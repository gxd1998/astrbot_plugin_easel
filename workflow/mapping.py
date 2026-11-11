# -*- coding: utf-8 -*-
"""Workflow graph slot mapping: infer, validate and materialize."""
from __future__ import annotations

import copy
from typing import Any

from ..utils import EaselError

_TEXT_NODE_TYPES = {
    "TextEncodeQwenImage21": ("prompt", "negative_prompt"),
    "CLIPTextEncode": ("text",),
    "CLIPTextEncodeSDXL": ("text_g", "text_l"),
}
_IMAGE_NODE_TYPES = {
    "LoadImage": "image",
    "ETN_LoadImageBase64": "image",
}
_VIDEO_NODE_TYPES = {
    "VHS_LoadVideo": "video",
}
_LATENT_TYPES = {"EmptyLatentImage"}
_SAMPLER_TYPES = {"KSampler", "KSamplerAdvanced"}
_KNOB_FIELDS = {"steps", "cfg"}


def infer(workflow: dict[str, Any]) -> dict[str, Any]:
    """Best-effort slot discovery; results are shown via /画图列表 for confirmation."""
    slots: dict[str, Any] = {"text": [], "image": [], "video": [], "latent_node": "", "sampler_node": ""}
    for node_id, node in workflow.items():
        class_type = node.get("class_type") or ""
        inputs = node.get("inputs") or {}
        if class_type in _TEXT_NODE_TYPES:
            for field_name in _TEXT_NODE_TYPES[class_type]:
                if field_name in inputs:
                    slots["text"].append({"node": node_id, "field": field_name})
        elif class_type in _IMAGE_NODE_TYPES and _IMAGE_NODE_TYPES[class_type] in inputs:
            slots["image"].append({"node": node_id, "field": _IMAGE_NODE_TYPES[class_type]})
        elif class_type in _VIDEO_NODE_TYPES and _VIDEO_NODE_TYPES[class_type] in inputs:
            slots["video"].append({"node": node_id, "field": _VIDEO_NODE_TYPES[class_type]})
        if not slots["latent_node"] and class_type in _LATENT_TYPES:
            slots["latent_node"] = node_id
        if not slots["sampler_node"] and class_type in _SAMPLER_TYPES:
            slots["sampler_node"] = node_id
    return slots


def validate(workflow: dict[str, Any], meta: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems; empty list means usable."""
    problems: list[str] = []
    slots = (meta or {}).get("slots") or {}
    for kind in ("text", "image", "video"):
        for slot in slots.get(kind) or []:
            node_id = str(slot.get("node") or "")
            field_name = slot.get("field") or ""
            if node_id not in workflow:
                problems.append(f"{kind} 槽指向不存在的节点 {node_id}")
            elif field_name not in (workflow[node_id].get("inputs") or {}):
                problems.append(f"{kind} 槽 {node_id}.{field_name} 字段不存在")
    return problems


def apply_request(
    workflow: dict[str, Any],
    meta: dict[str, Any],
    *,
    request,
    uploads: dict[str, str],
    width: int,
    height: int,
    seed: int | None,
    knobs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Deep-copy the graph and inject one request's values into every slot."""
    graph = copy.deepcopy(workflow)
    slots = (meta or {}).get("slots") or {}

    prompt_injected = False
    for slot in slots.get("text") or []:
        node_id, field_name = str(slot.get("node")), str(slot.get("field") or "")
        if field_name == "negative_prompt":
            _set_field(graph, node_id, field_name, request.negative)
        elif not prompt_injected:
            _set_field(graph, node_id, field_name, request.prompt)
            prompt_injected = True

    image_slots = slots.get("image") or []
    for index, slot in enumerate(image_slots):
        node_id, field_name = str(slot.get("node")), str(slot.get("field") or "")
        feed_mode = slot.get("feed_mode") or "auto"
        if feed_mode == "fixed":
            _set_field(graph, node_id, field_name, slot.get("fixed_name") or "")
        elif str(index) in uploads:
            _set_field(graph, node_id, field_name, uploads[str(index)])

    for slot in slots.get("video") or []:
        node_id, field_name = str(slot.get("node")), str(slot.get("field") or "")
        _set_field(graph, node_id, field_name, slot.get("fixed_name") or "")

    latent_node = str(meta.get("latent_node") or "")
    if latent_node and latent_node in graph:
        inputs = graph[latent_node].setdefault("inputs", {})
        if "width" in inputs:
            inputs["width"] = width
        if "height" in inputs:
            inputs["height"] = height

    sampler_node = str(meta.get("sampler_node") or "")
    if sampler_node and sampler_node in graph:
        inputs = graph[sampler_node].setdefault("inputs", {})
        if seed is not None and "seed" in inputs:
            inputs["seed"] = seed
        for key, value in (knobs or {}).items():
            if key in _KNOB_FIELDS and key in inputs:
                inputs[key] = value

    if seed is not None and "seed" not in (graph.get(sampler_node, {}).get("inputs") or {}):
        # KSamplerAdvanced uses noise_seed instead.
        for node_id, node in graph.items():
            inputs = node.get("inputs") or {}
            if node.get("class_type") in _SAMPLER_TYPES and "noise_seed" in inputs:
                inputs["noise_seed"] = seed
                break
    return graph


def inject_lora(graph: dict[str, Any], lora_spec: dict[str, Any] | None) -> None:
    """Materialize a workflow's LoRA node from the (config-merged) spec."""
    if not isinstance(lora_spec, dict) or not str(lora_spec.get("node") or ""):
        return
    node_id = str(lora_spec["node"])
    name = str(lora_spec.get("name") or "").strip()
    if not name:
        raise EaselError(
            "该工作流需要 LoRA：请在插件设置 workflows 里给它配置 lora.name"
            "（文件需位于 ComfyUI 的 models/loras/ 下，且与底模架构兼容）"
        )
    if node_id not in graph:
        raise EaselError(f"LoRA 注入点节点 {node_id} 不在工作流中")
    inputs = graph[node_id].setdefault("inputs", {})
    inputs["lora_name"] = name
    strength = float(lora_spec.get("strength_model", 1.0))
    if "strength_model" in inputs:
        inputs["strength_model"] = strength
    if "strength_clip" in inputs:
        inputs["strength_clip"] = strength


def _set_field(graph: dict[str, Any], node_id: str, field_name: str, value: Any) -> None:
    if node_id not in graph:
        raise EaselError(f"工作流节点 {node_id} 不存在")
    inputs = graph[node_id].setdefault("inputs", {})
    if field_name not in inputs:
        raise EaselError(f"工作流节点 {node_id} 不含字段 {field_name}")
    inputs[field_name] = value

# -*- coding: utf-8 -*-
"""Offline unit tests for the workflow mapping layer (no AstrBot runtime needed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from astrbot_plugin_easel.models import MediaRequest  # noqa: E402
from astrbot_plugin_easel.workflow import mapping  # noqa: E402

BUILTIN = Path(__file__).resolve().parent.parent / "workflow" / "builtin"


def load_workflow(name: str) -> dict:
    return json.loads((BUILTIN / f"{name}.json").read_text(encoding="utf-8-sig"))


def load_meta(name: str) -> dict:
    return json.loads((BUILTIN / f"{name}.meta.json").read_text(encoding="utf-8-sig"))


def test_builtin_workflows_validate():
    for name in ("qwen_t2i", "qwen_t2i_ref"):
        problems = mapping.validate(load_workflow(name), load_meta(name))
        assert problems == [], f"{name}: {problems}"


def test_infer_finds_text_and_image_slots():
    slots = mapping.infer(load_workflow("qwen_t2i_ref"))
    text_fields = [(s["node"], s["field"]) for s in slots["text"]]
    assert ("5", "prompt") in text_fields and ("5", "negative_prompt") in text_fields
    assert slots["image"] == [{"node": "12", "field": "image"}]
    assert slots["latent_node"] == "8" and slots["sampler_node"] == "3"


def test_apply_request_injects_everything():
    graph = mapping.apply_request(
        load_workflow("qwen_t2i_ref"),
        load_meta("qwen_t2i_ref"),
        request=MediaRequest(prompt="一只鲸鱼", negative="低质量"),
        uploads={"0": "uploaded_ref.png"},
        width=768,
        height=1344,
        seed=123456,
        knobs={"steps": 20, "cfg": 1.0},
    )
    assert graph["5"]["inputs"]["prompt"] == "一只鲸鱼"
    assert graph["5"]["inputs"]["negative_prompt"] == "低质量"
    assert graph["5"]["inputs"]["images.image_1"] == ["12", 0]
    assert graph["12"]["inputs"]["image"] == "uploaded_ref.png"
    assert graph["8"]["inputs"]["width"] == 768
    assert graph["8"]["inputs"]["height"] == 1344
    assert graph["3"]["inputs"]["seed"] == 123456
    # original workflow must stay untouched (deep copy semantics)
    assert load_workflow("qwen_t2i_ref")["3"]["inputs"]["seed"] == 0


def test_fixed_feed_mode_uses_fixed_name():
    meta = load_meta("qwen_t2i_ref")
    meta["slots"]["image"][0]["feed_mode"] = "fixed"
    meta["slots"]["image"][0]["fixed_name"] = "persona.png"
    graph = mapping.apply_request(
        load_workflow("qwen_t2i_ref"), meta,
        request=MediaRequest(prompt="x"), uploads={}, width=1024, height=1024, seed=1, knobs={},
    )
    assert graph["12"]["inputs"]["image"] == "persona.png"


def test_missing_node_raises():
    meta = load_meta("qwen_t2i")
    meta["slots"]["text"][0]["node"] = "999"
    try:
        mapping.apply_request(
            load_workflow("qwen_t2i"), meta,
            request=MediaRequest(prompt="x"), uploads={}, width=1024, height=1024, seed=1, knobs={},
        )
    except Exception as exc:  # noqa: BLE001
        assert "999" in str(exc)
    else:
        raise AssertionError("expected EaselError for missing node")


def test_inject_lora_materializes_node():
    graph = mapping.apply_request(
        load_workflow("qwen_t2i_lora"), load_meta("qwen_t2i_lora"),
        request=MediaRequest(prompt="x"), uploads={}, width=1024, height=1024, seed=7, knobs={},
    )
    mapping.inject_lora(graph, {"node": "13", "name": "role_lora.safetensors", "strength_model": 0.8})
    assert graph["13"]["inputs"]["lora_name"] == "role_lora.safetensors"
    assert graph["13"]["inputs"]["strength_model"] == 0.8
    assert graph["3"]["inputs"]["model"] == ["13", 0]


def test_inject_lora_requires_name():
    graph = mapping.apply_request(
        load_workflow("qwen_t2i_lora"), load_meta("qwen_t2i_lora"),
        request=MediaRequest(prompt="x"), uploads={}, width=1024, height=1024, seed=7, knobs={},
    )
    try:
        mapping.inject_lora(graph, {"node": "13", "name": ""})
    except Exception as exc:  # noqa: BLE001
        assert "lora.name" in str(exc)
    else:
        raise AssertionError("expected EaselError when lora.name is empty")


def test_role_slots_and_prompt_suffix():
    meta = load_meta("whalemaid_emoji")
    problems = mapping.validate(load_workflow("whalemaid_emoji"), meta)
    assert problems == [], problems
    graph = mapping.apply_request(
        load_workflow("whalemaid_emoji"), meta,
        request=MediaRequest(prompt="在吃饭，鼓着腮帮子", negative="多余文字"),
        uploads={}, width=1024, height=1024, seed=9, knobs={},
    )
    pos = graph["5"]["inputs"]["text"]
    assert pos.startswith("masterpiece") and "在吃饭，鼓着腮帮子" in pos and "ds_whalemaid" in pos and "maid outfit" in pos
    assert "多余文字" in graph["6"]["inputs"]["text"] and "worst quality" in graph["6"]["inputs"]["text"]
    mapping.inject_lora(graph, meta["lora"])
    assert graph["2"]["inputs"]["lora_name"] == "ds_whalemaid.safetensors"
    assert graph["8"]["inputs"]["model"] == ["2", 0]


if __name__ == "__main__":
    test_builtin_workflows_validate()
    test_infer_finds_text_and_image_slots()
    test_apply_request_injects_everything()
    test_fixed_feed_mode_uses_fixed_name()
    test_missing_node_raises()
    test_inject_lora_materializes_node()
    test_inject_lora_requires_name()
    test_role_slots_and_prompt_suffix()
    print("8/8 mapping tests passed")

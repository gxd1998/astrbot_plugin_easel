# -*- coding: utf-8 -*-
"""Offline unit tests for the workflow mapping layer (no AstrBot runtime needed)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from astrbot_plugin_easel.models import MediaRequest  # noqa: E402
from astrbot_plugin_easel.workflow import mapping  # noqa: E402
from astrbot_plugin_easel.workflow.store import WorkflowStore  # noqa: E402

BUILTIN = Path(__file__).resolve().parent.parent / "workflow" / "builtin"


def load_workflow(name: str) -> dict:
    return json.loads((BUILTIN / f"{name}.json").read_text(encoding="utf-8-sig"))


def load_meta(name: str) -> dict:
    return json.loads((BUILTIN / f"{name}.meta.json").read_text(encoding="utf-8-sig"))


def test_builtin_workflows_validate():
    for name in ("qwen_t2i", "qwen_t2i_ref", "whalegirl", "whalemaid_emoji", "doubao"):
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
        load_workflow("whalegirl"), load_meta("whalegirl"),
        request=MediaRequest(prompt="x"), uploads={}, width=1024, height=1024, seed=7, knobs={},
    )
    mapping.inject_lora(graph, {"node": "13", "name": "role_lora.safetensors", "strength_model": 0.8})
    assert graph["13"]["inputs"]["lora_name"] == "role_lora.safetensors"
    assert graph["13"]["inputs"]["strength_model"] == 0.8
    assert graph["3"]["inputs"]["model"] == ["13", 0]


def test_inject_lora_requires_name():
    graph = mapping.apply_request(
        load_workflow("whalegirl"), load_meta("whalegirl"),
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


def test_whalegirl_workflow():
    meta = load_meta("whalegirl")
    assert "女仆" not in meta["prompt_prefix"]  # 换装前提：prefix 只锁身体特征，不含服装
    problems = mapping.validate(load_workflow("whalegirl"), meta)
    assert problems == [], problems
    graph = mapping.apply_request(
        load_workflow("whalegirl"), meta,
        request=MediaRequest(prompt="穿着黑色蕾丝睡衣坐在床边"),
        uploads={}, width=1024, height=1024, seed=9, knobs={},
    )
    pos = graph["5"]["inputs"]["prompt"]
    assert pos.startswith("年轻鲸鱼娘") and "穿着黑色蕾丝睡衣坐在床边" in pos
    assert pos.endswith("二次元插画风格，干净的线条，精致的细节，高质量")
    mapping.inject_lora(graph, meta["lora"])
    assert graph["13"]["inputs"]["lora_name"] == "whalegirl_young_qwen21.safetensors"
    assert graph["13"]["inputs"]["strength_model"] == 0.6
    assert graph["3"]["inputs"]["model"] == ["13", 0]


def test_doubao_workflow():
    """豆包没有 LoRA：工作流里绝不能残留 LoRA 节点，否则会继承鲸鱼娘的身份。"""
    workflow = load_workflow("doubao")
    meta = load_meta("doubao")
    assert mapping.validate(workflow, meta) == []
    assert all(node["class_type"] != "LoraLoaderModelOnly" for node in workflow.values())
    assert workflow["3"]["inputs"]["model"] == ["4", 0]
    assert "lora" not in meta
    # 前缀只锁长相，不含服装（换装前提）
    for outfit_word in ("围巾", "西装", "短裙", "靴"):
        assert outfit_word not in meta["prompt_prefix"], outfit_word
    assert meta["default_size"] == "3:4"
    graph = mapping.apply_request(
        workflow, meta,
        request=MediaRequest(prompt="红色粗针织围巾，站在便利店门口"),
        uploads={}, width=960, height=1280, seed=9, knobs={},
    )
    pos = graph["5"]["inputs"]["prompt"]
    assert pos.startswith("一位二十岁出头的东亚年轻女性")
    assert "栗棕色齐下巴波波头" in pos
    assert "红色粗针织围巾，站在便利店门口" in pos
    assert pos.endswith("超写实，8K")
    assert "写实电影感人物摄影" in pos


def test_character_lookup():
    store = WorkflowStore(BUILTIN, Path(__file__).resolve().parent / "_no_user_workflows")
    assert store.find_character("豆包").name == "doubao"
    assert store.find_character("豆包酱").name == "doubao"      # 别名
    assert store.find_character("DOUBAO").name == "doubao"      # 大小写无关
    assert store.find_character("画个豆包").name == "doubao"    # 子串兜底
    assert store.find_character("鲸鱼娘").name == "whalegirl"
    assert store.find_character("大肥鱼").name == "whalegirl"
    assert store.find_character("") is None
    assert store.find_character("不存在的角色") is None
    # 表情包工作流不参与角色路由，只能按 workflow_name 显式指定
    assert store.get("whalemaid_emoji").character == ""
    assert set(store.characters()) == {"豆包", "鲸鱼娘"}
    row = next(r for r in store.summaries() if r["name"] == "doubao")
    assert row["character"] == "豆包" and "豆包酱" in row["aliases"]


if __name__ == "__main__":
    test_builtin_workflows_validate()
    test_infer_finds_text_and_image_slots()
    test_apply_request_injects_everything()
    test_fixed_feed_mode_uses_fixed_name()
    test_missing_node_raises()
    test_inject_lora_materializes_node()
    test_inject_lora_requires_name()
    test_role_slots_and_prompt_suffix()
    test_whalegirl_workflow()
    test_doubao_workflow()
    test_character_lookup()
    print("11/11 mapping tests passed")

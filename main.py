# -*- coding: utf-8 -*-
"""Easel — a standalone image-generation plugin for AstrBot.

ComfyUI-native workflows plus OpenAI-compatible online providers, exposed both
as LLM tools (the model decides how to deliver) and as explicit commands.
"""
from __future__ import annotations

import asyncio
import json
import shlex
import time
from collections import deque
from pathlib import Path
from typing import Any

import aiohttp

from astrbot.api import logger, llm_tool
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools, register
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.star.filter.permission import PermissionType
from astrbot.core.utils.astrbot_path import get_astrbot_temp_path

from .backends.ark_images import ArkImagesBackend
from .backends.comfyui import ComfyUIBackend
from .backends.custom_http import CustomHTTPBackend
from .backends.openai_images import OpenAIImagesBackend
from .models import MediaRequest, MediaResult
from .utils import EaselError, parse_size
from .workflow.store import WorkflowStore

PLUGIN_NAME = "astrbot_plugin_easel"


@register(
    PLUGIN_NAME,
    "vamos",
    "画架 Easel — 独立生图插件（ComfyUI 工作流 + 在线生图，LLM 工具与命令双触发）",
    "0.1.0",
    "https://github.com/vamos/astrbot_plugin_easel",
)
class EaselPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None):
        super().__init__(context)
        self.config = config if config is not None else {}
        self._store: WorkflowStore | None = None
        self._backends: dict[str, Any] = {}
        self._history: deque[dict[str, Any]] = deque(maxlen=20)
        self._cleanup_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        data_dir = Path(StarTools.get_data_dir(PLUGIN_NAME))
        out_dir = Path(get_astrbot_temp_path()) / PLUGIN_NAME.rsplit("plugin_", 1)[-1]
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._store = WorkflowStore(
            builtin_dir=Path(__file__).parent / "workflow" / "builtin",
            user_dir=data_dir / "workflows",
        )
        self._store.set_overrides(self.config.get("workflows") or {})
        self._backends = {
            "comfyui": ComfyUIBackend(self.config.get("comfyui") or {}, self._store, out_dir),
        }
        ark_cfg = self.config.get("ark_images") or {}
        if ark_cfg.get("enabled"):
            self._backends["ark_images"] = ArkImagesBackend(ark_cfg, out_dir)
        oai_cfg = self.config.get("openai_images") or {}
        if oai_cfg.get("enabled"):
            self._backends["openai_images"] = OpenAIImagesBackend(oai_cfg, out_dir)
        for name, entry in (self.config.get("custom_backends") or {}).items():
            if isinstance(entry, dict):
                self._backends[f"custom:{name}"] = CustomHTTPBackend(name, entry, out_dir)
        keep_days = int((self.config.get("output") or {}).get("keep_days") or 0)
        if keep_days:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop(keep_days))
        workflows = ", ".join(self._store.names()) or "（无）"
        logger.info(f"[Easel] 初始化完成：工作流 {workflows}")

    async def terminate(self) -> None:
        if self._cleanup_task:
            self._cleanup_task.cancel()
        for backend in self._backends.values():
            await backend.close()
        logger.info("[Easel] 已退出")

    # ------------------------------------------------------------------
    # LLM tools
    # ------------------------------------------------------------------
    @llm_tool(name="paint_generate")
    async def tool_paint_generate(
        self,
        event: AstrMessageEvent,
        prompt: str,
        workflow_name: str = "",
        size: str = "",
        reference_image: str = "",
        use_persona: bool = False,
    ) -> str:
        """生成一张图片。当用户想要画图、生成图片、画一个某物、来一张图时调用；
        用户带着图片或引用一条带图消息并要求以此为参考/改图时也调用并传入 reference_image。

        图片生成后不会自动发出：结果 JSON 里有本地 path，请用内置 send_message_to_user
        工具（messages=[{"type":"image","path":"<path>"}]）发送，或按对话情境改用文字回应。
        若 ok=false，把 error 中的原因告诉用户，不要声称已生成。

        Args:
            prompt(string): 画面描述，写具体可见的细节，不要抽象指令
            workflow_name(string): 工作流名，留空使用默认；可先用 paint_list_workflows 查询
            size(string): 画幅：1:1 / 3:4 / 4:3 / 9:16 / 16:9 或 1024x1024，留空默认 1:1
            reference_image(string): 参考图本地路径或 URL；仅当用户消息带图或引用了带图消息时传
            use_persona(boolean): true 时按固定人设出图（使用配置的 persona 参考图与工作流）
        """
        result = await self._generate(prompt=prompt, workflow_name=workflow_name, size=size,
                                      reference_image=reference_image, use_persona=use_persona)
        self._record(result)
        return json.dumps(result.to_json(), ensure_ascii=False)

    @llm_tool(name="paint_list_workflows")
    async def tool_paint_list_workflows(self, event: AstrMessageEvent) -> str:
        """查询当前可用的生图工作流列表。

        当用户询问能画什么、有哪些画风或工作流，或需要为 paint_generate 选择
        workflow_name 时调用。返回工作流名、说明与是否支持参考图。"""
        rows = []
        for row in self._store.summaries():
            flags = []
            if (row.get("slots") or {}).get("image"):
                flags.append("支持参考图")
            rows.append(f"{row['name']}（{row['source']}）：{row['description'] or '-'}"
                        + ("；" + "、".join(flags) if flags else "")
                        + (f"；问题：{';'.join(row['problems'])}" if row["problems"] else ""))
        return "\n".join(rows) if rows else "（尚未注册任何工作流）"

    # ------------------------------------------------------------------
    # commands
    # ------------------------------------------------------------------
    @filter.command("画", alias={"画图"})
    async def command_paint(self, event: AstrMessageEvent, *args: str) -> None:
        """/画 [参数] 描述 —— 命令兜底：直接生成并立即发送，不经过 LLM。"""
        if not self._allowed(event):
            yield event.plain_result("🚫 当前账号不在画架白名单内")
            return
        try:
            options, prompt = self._parse_paint_args(args)
        except EaselError as exc:
            yield event.plain_result(f"❌ {exc}")
            return
        if not prompt:
            yield event.plain_result(
                "用法：/画 [--wf=工作流名] [--size=1:1] [--seed=数字] [--ref=参考图路径] 描述"
            )
            return
        result = await self._generate(prompt=prompt, **options)
        self._record(result)
        if result.ok:
            yield event.image_result(str(result.files[0]))
        else:
            yield event.plain_result(f"❌ 生成失败：{result.error}")

    @filter.command("画图列表")
    async def command_workflow_list(self, event: AstrMessageEvent) -> None:
        if not self._allowed(event):
            yield event.plain_result("🚫 当前账号不在画架白名单内")
            return
        lines = ["🎨 工作流列表："]
        for row in self._store.summaries():
            state = "✅" if not row["problems"] else "⚠️"
            lines.append(
                f"{state} {row['name']} [{row['source']}] {row['description']}"
                + (f" ｜⚠ {';'.join(row['problems'])}" if row["problems"] else "")
            )
        yield event.plain_result("\n".join(lines))

    @filter.command("画图状态")
    async def command_status(self, event: AstrMessageEvent) -> None:
        if not self._allowed(event):
            yield event.plain_result("🚫 当前账号不在画架白名单内")
            return
        lines = ["📊 最近生成："]
        if not self._history:
            lines.append("（暂无记录）")
        for item in list(self._history)[-5:]:
            mark = "✅" if item["ok"] else "❌"
            detail = item.get("files", [])[0] if item["ok"] else item.get("error", "")[:80]
            lines.append(f"{mark} {item['workflow']}@{item['backend']} {item['seconds']:.1f}s ｜ {detail}")
        for name, backend in self._backends.items():
            health = await backend.health()
            mark = "✅" if health.get("ok") else "❌"
            extra = ""
            if name == "comfyui" and health.get("ok"):
                extra = f"（显存余量 {health['vram_free_mb']}MB，{health['workflows']} 个工作流）"
            elif name != "comfyui" and health.get("ok"):
                extra = f"（{health.get('model', '')}）"
            lines.append(f"{mark} {name}：{'正常' + extra if health.get('ok') else health.get('error', '不可达')}")
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("画图测试")
    async def command_test(self, event: AstrMessageEvent) -> None:
        lines = ["🧪 画架自检："]
        for name, backend in self._backends.items():
            health = await backend.health()
            mark = "✅" if health.get("ok") else "❌"
            lines.append(f"{mark} {name}：{json.dumps(health, ensure_ascii=False)}")
        yield event.plain_result("\n".join(lines))

    @filter.permission_type(PermissionType.ADMIN)
    @filter.command("画图重载")
    async def command_reload(self, event: AstrMessageEvent) -> None:
        self._store.reload()
        yield event.plain_result(f"🔄 已重载：{', '.join(self._store.names()) or '（无）'}")

    # ------------------------------------------------------------------
    # generation pipeline
    # ------------------------------------------------------------------
    async def _generate(
        self,
        *,
        prompt: str,
        workflow_name: str = "",
        size: str = "",
        reference_image: str = "",
        use_persona: bool = False,
        seed: int | None = None,
        negative: str = "",
        backend: str = "",
    ) -> MediaResult:
        if not bool(self.config.get("enable", True)):
            return MediaResult(ok=False, error="插件未启用")
        prompt = (prompt or "").strip()
        if not prompt:
            return MediaResult(ok=False, error="画面描述为空")
        if use_persona:
            persona = self.config.get("persona") or {}
            persona_ref = str(persona.get("reference_path") or "").strip()
            if not persona_ref:
                return MediaResult(ok=False, error="固定人设参考图未配置：请在插件设置 persona.reference_path 里填写服务器上的图片路径")
            workflow_name = str(persona.get("workflow") or "qwen_t2i_ref")
            reference_image = persona_ref
            extra = str(persona.get("extra_prompt") or "").strip()
            if extra and extra.lower() not in prompt.lower():
                prompt = f"{prompt}，{extra}"
        try:
            workflow_def = self._resolve_workflow(workflow_name, reference_image)
        except EaselError as exc:
            return MediaResult(ok=False, error=str(exc))

        presets = self.config.get("size_presets") or {}
        effective_size = size or workflow_def.default_size
        try:
            width, height = parse_size(effective_size, presets, default=workflow_def.default_size)
        except EaselError as exc:
            return MediaResult(ok=False, error=str(exc))

        images: list[Path] = []
        if reference_image:
            images = await self._resolve_reference(reference_image)
            if not images:
                return MediaResult(ok=False, error=f"参考图不可用：{reference_image}")

        request = MediaRequest(
            prompt=prompt,
            negative=negative or "",
            size=effective_size,
            width=width,
            height=height,
            images=images,
            workflow_name=workflow_def.name,
            seed=seed,
        )
        return await self._run_chain(request, backend_hint=backend)

    async def _run_chain(self, request: MediaRequest, *, backend_hint: str = "") -> MediaResult:
        """Try the default backend first, then the fallback chain; first win returns.

        A non-empty backend_hint takes priority (e.g. /画 --backend=ark_images).
        """
        default = str(backend_hint or self.config.get("default_backend") or "comfyui")
        chain = [default] + [b for b in (self.config.get("fallback_chain") or []) if b != default]
        notes: list[str] = []
        for name in chain:
            backend = self._backends.get(name)
            if backend is None:
                notes.append(f"{name}：未启用")
                continue
            result = await backend.generate(request)
            if result.ok:
                return result
            notes.append(f"{name}：{result.error}")
        last_error = notes[-1] if notes else "无可用后端"
        return MediaResult(ok=False, backend=default, workflow=request.workflow_name,
                           error=f"所有后端均失败（{'; '.join(notes) or last_error}）")

    def _resolve_workflow(self, name: str, reference_image: str):
        store = self._store
        definition = store.get(name) if name else None
        if definition is None and name:
            raise EaselError(f"工作流「{name}」不存在。可用：{', '.join(store.names())}")
        if definition is None:
            definition = store.default()
            if definition is None:
                raise EaselError("尚未注册任何工作流")
        if reference_image and not (definition.slots or {}).get("image"):
            fallback = next(
                (row for row in (store.get(n) for n in store.names())
                 if row and (row.slots or {}).get("image")),
                None,
            )
            if fallback is not None:
                definition = fallback
        return definition

    async def _resolve_reference(self, reference: str) -> list[Path]:
        reference = (reference or "").strip()
        if not reference:
            return []
        if reference.startswith("http://") or reference.startswith("https://"):
            target = self._out_dir / f"ref_{asyncio.get_event_loop().time():.0f}.png"
            try:
                async with aiohttp.ClientSession() as http:
                    async with http.get(reference) as resp:
                        if resp.status != 200:
                            return []
                        target.write_bytes(await resp.read())
            except Exception:  # noqa: BLE001
                return []
            return [target]
        path = Path(reference)
        return [path] if path.is_file() else []

    def _backend(self, name: str) -> Any:
        return self._backends.get(name)

    # ------------------------------------------------------------------
    # access control / bookkeeping
    # ------------------------------------------------------------------
    def _allowed(self, event: AstrMessageEvent) -> bool:
        whitelist = list((self.config.get("access") or {}).get("command_whitelist") or [])
        if not whitelist:
            return True
        whitelist = [str(x) for x in whitelist]
        uid = str(event.get_sender_id() or "")
        gid = str(event.get_group_id() or "")
        return uid in whitelist or gid in whitelist or f"{gid}:{uid}" in whitelist

    @staticmethod
    def _parse_paint_args(args: tuple[str, ...]) -> tuple[dict[str, Any], str]:
        options: dict[str, Any] = {}
        text_parts: list[str] = []
        try:
            tokens = shlex.split(" ".join(args), posix=True)
        except ValueError as exc:
            raise EaselError(f"参数解析失败：{exc}") from exc
        key_map = {
            "wf": "workflow_name", "workflow": "workflow_name",
            "size": "size", "neg": "negative", "negative": "negative",
            "seed": "seed", "ref": "reference_image", "reference": "reference_image",
            "persona": "use_persona", "be": "backend", "backend": "backend",
        }
        bool_flags = {"use_persona"}
        for token in tokens:
            if token.startswith("--"):
                key, _, value = token[2:].partition("=")
                mapped = key_map.get(key)
                if not mapped:
                    raise EaselError(f"未知参数 --{key}")
                if not value:
                    if mapped in bool_flags:
                        value = "1"
                    else:
                        raise EaselError(f"参数 --{key} 缺少值")
                if mapped in bool_flags:
                    options[mapped] = value.strip().lower() not in {"0", "false", "no", "off", "否"}
                else:
                    options[mapped] = value
            else:
                text_parts.append(token)
        if "seed" in options:
            try:
                options["seed"] = int(options["seed"])
            except ValueError as exc:
                raise EaselError("seed 必须是整数") from exc
        return options, " ".join(text_parts).strip()

    def _record(self, result: MediaResult) -> None:
        self._history.append({
            "ok": result.ok,
            "workflow": result.workflow,
            "backend": result.backend,
            "seconds": result.seconds,
            "files": [str(p) for p in result.files],
            "error": result.error,
        })

    async def _cleanup_loop(self, keep_days: int) -> None:
        while True:
            try:
                await asyncio.sleep(6 * 3600)
                now = time.time()
                for path in self._out_dir.glob("*"):
                    if path.is_file() and now - path.stat().st_mtime > keep_days * 86400:
                        path.unlink(missing_ok=True)
            except asyncio.CancelledError:
                return
            except Exception as exc:  # noqa: BLE001 - maintenance must not die
                logger.warning(f"[Easel] 清理任务异常：{exc}")

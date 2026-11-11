# -*- coding: utf-8 -*-
"""Native ComfyUI backend: submit API workflows, poll, download, upload references."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import aiohttp

from ..models import MediaRequest, MediaResult
from ..utils import EaselError, humanize_error, monotonic, output_stem, random_seed
from ..workflow.mapping import apply_request, inject_lora
from ..workflow.store import WorkflowStore

_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                   ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp"}


class ComfyUIBackend:
    """Runs ComfyUI API-format workflows against one shared server."""

    name = "comfyui"
    kind = "image"

    def __init__(self, config: dict[str, Any], store: WorkflowStore, out_dir: Path) -> None:
        self._cfg = config
        self._store = store
        self._out_dir = out_dir
        self._session: aiohttp.ClientSession | None = None
        self._upload_cache: dict[str, str] = {}

    # -- transport ---------------------------------------------------------
    def _base(self) -> str:
        base = str(self._cfg.get("base_url") or "").strip().rstrip("/")
        if not base:
            raise EaselError("ComfyUI base_url 未配置，请在插件设置中填写服务地址")
        if "://" not in base:
            base = "http://" + base
        return base

    async def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=int(self._cfg.get("timeout_seconds", 180)) + 30),
            )
        return self._session

    # -- health ------------------------------------------------------------
    async def health(self) -> dict:
        base = self._base()
        http = await self._http()
        try:
            async with http.get(f"{base}/system_stats") as resp:
                if resp.status != 200:
                    return {"ok": False, "error": f"HTTP {resp.status}"}
                data = await resp.json()
            devices = (data.get("devices") or [{}])[0]
            return {
                "ok": True,
                "version": data.get("system", {}).get("comfyui_version", ""),
                "vram_free_mb": int(devices.get("vram_free", 0) // 1024 // 1024),
                "workflows": len(self._store.names()),
            }
        except Exception as exc:  # noqa: BLE001 - diagnostics must not raise
            return {"ok": False, "error": humanize_error(exc, self.name)}

    # -- generation --------------------------------------------------------
    async def generate(self, request: MediaRequest) -> MediaResult:
        started = monotonic()
        try:
            definition = self._store.get(request.workflow_name)
            if definition is None:
                if request.workflow_name:
                    raise EaselError(
                        f"工作流「{request.workflow_name}」不存在，可用：{', '.join(self._store.names()) or '（空）'}"
                    )
                definition = self._store.default()
                if definition is None:
                    raise EaselError("尚未注册任何工作流")
                if request.images and not (definition.slots or {}).get("image"):
                    # Reference images need a workflow that can accept them.
                    fallback = next(
                        (d for n in self._store.names()
                         for d in [self._store.get(n)]
                         if d and (d.slots or {}).get("image")),
                        None,
                    )
                    if fallback is not None:
                        definition = fallback

            graph = await self._build_graph(definition, request)
            base = self._base()
            http = await self._http()
            payload = {"prompt": graph, "client_id": str(self._cfg.get("client_id") or "astrbot-plugin-easel")}
            async with http.post(f"{base}/prompt", json=payload) as resp:
                body = await resp.json(content_type=None)
                if resp.status != 200:
                    raise EaselError(self._translate_submit_error(body, resp.status))
            prompt_id = str(body.get("prompt_id") or "")
            if not prompt_id:
                raise EaselError(f"ComfyUI 未返回任务 ID：{body}")

            files = await self._poll_and_fetch(prompt_id, definition.name)
            return MediaResult(
                ok=True,
                files=files,
                backend=self.name,
                workflow=definition.name,
                seconds=monotonic() - started,
                width=request.width or 0,
                height=request.height or 0,
            )
        except EaselError as exc:
            return MediaResult(ok=False, backend=self.name, workflow=request.workflow_name,
                               seconds=monotonic() - started, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - caller expects a result, not a raise
            return MediaResult(ok=False, backend=self.name, workflow=request.workflow_name,
                               seconds=monotonic() - started, error=humanize_error(exc, self.name))

    async def _build_graph(self, definition, request: MediaRequest) -> dict[str, Any]:
        """Materialize the definition: uploads, slots, size, seed, sampler knobs."""
        meta = {
            "slots": definition.slots,
            "latent_node": definition.latent_node,
            "sampler_node": definition.sampler_node,
            "seed_mode": definition.seed_mode,
        }
        uploaded: dict[str, str] = {}
        for index, image_path in enumerate(request.images):
            uploaded[str(index)] = await self._upload_image(image_path)

        width = request.width or 0
        height = request.height or 0
        if not (width and height):
            from ..utils import parse_size

            width, height = parse_size(request.size, {}, default=definition.default_size)
        request.width, request.height = width, height

        seed = request.seed if request.seed is not None else (
            random_seed() if meta["seed_mode"] == "random" else None
        )
        knobs = {
            "steps": self._cfg.get("steps"),
            "cfg": self._cfg.get("cfg"),
        }
        graph = apply_request(
            definition.workflow,
            meta,
            request=request,
            uploads=uploaded,
            width=width,
            height=height,
            seed=seed,
            knobs={k: v for k, v in knobs.items() if v is not None},
        )
        inject_lora(graph, definition.lora)
        return graph

    async def _upload_image(self, path: Path) -> str:
        """POST /upload/image and return the server-side filename (memoized per file)."""
        if not path or not Path(path).exists():
            raise EaselError(f"参考图不存在：{path}")
        stat = Path(path).stat()
        cache_key = f"{Path(path).resolve()}|{stat.st_size}|{int(stat.st_mtime)}"
        cached = self._upload_cache.get(cache_key)
        if cached:
            return cached
        suffix = Path(path).suffix.lower() or ".png"
        mime = _MIME_BY_SUFFIX.get(suffix, "image/png")
        remote_name = f"easel_{output_stem('ref')}{suffix}"
        http = await self._http()
        base = self._base()
        form = aiohttp.FormData()
        form.add_field("image", Path(path).read_bytes(), filename=remote_name, content_type=mime)
        form.add_field("overwrite", "false")
        form.add_field("type", "input")
        async with http.post(f"{base}/upload/image", data=form) as resp:
            body = await resp.json(content_type=None)
            if resp.status != 200:
                raise EaselError(f"参考图上传失败：HTTP {resp.status} {body}")
        name = str(body.get("name") or "")
        if not name:
            raise EaselError(f"参考图上传未返回文件名：{body}")
        self._upload_cache[cache_key] = name
        return name

    async def _poll_and_fetch(self, prompt_id: str, workflow_name: str) -> list[Path]:
        base = self._base()
        http = await self._http()
        interval = float(self._cfg.get("poll_interval") or 1.0)
        deadline = monotonic() + float(self._cfg.get("timeout_seconds") or 180)
        while monotonic() < deadline:
            async with http.get(f"{base}/history/{prompt_id}") as resp:
                history = await resp.json(content_type=None)
            entry = (history or {}).get(prompt_id)
            if entry:
                status = (entry.get("status") or {})
                if status.get("status_str") == "error":
                    raise EaselError(self._translate_history_error(entry))
                outputs = entry.get("outputs") or {}
                files = await self._fetch_outputs(outputs, workflow_name)
                if files:
                    return files
            await asyncio.sleep(interval)
        raise EaselError(f"ComfyUI 任务 {prompt_id[:8]} 超时未完成（timeout_seconds 内无输出）")

    async def _fetch_outputs(self, outputs: dict[str, Any], workflow_name: str = "easel") -> list[Path]:
        base = self._base()
        http = await self._http()
        files: list[Path] = []
        for node_output in outputs.values():
            for image in node_output.get("images") or []:
                filename = image.get("filename") or ""
                if not filename:
                    continue
                params = {"filename": filename, "type": image.get("type") or "output",
                          "subfolder": image.get("subfolder") or ""}
                async with http.get(f"{base}/view", params=params) as resp:
                    if resp.status != 200:
                        raise EaselError(f"取回图片失败：HTTP {resp.status}")
                    blob = await resp.read()
                out_path = self._out_dir / f"{output_stem(workflow_name)}_{len(files)}_{Path(filename).suffix or '.png'}"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(blob)
                files.append(out_path)
        return files

    # -- error translation -------------------------------------------------
    @staticmethod
    def _translate_submit_error(body: Any, status: int) -> str:
        node_errors = (body or {}).get("node_errors") or {}
        if not node_errors:
            return f"ComfyUI 提交被拒（HTTP {status}）：{str(body)[:300]}"
        first_node, detail = next(iter(node_errors.items()))
        errors = (detail or {}).get("errors") or []
        parts = []
        for item in errors:
            field = ((item.get("extra_info") or {}).get("input_name")) or item.get("details") or ""
            message = item.get("message") or ""
            parts.append(f"节点{first_node} {field}: {message}")
        hint = "；".join(parts) or str(detail)[:200]
        if "value not in list" in hint.lower():
            hint += " —— 通常表示工作流里的模型/文件名在本机 ComfyUI 中不存在"
        return f"ComfyUI 提交被拒（HTTP {status}）：{hint}"

    @staticmethod
    def _translate_history_error(entry: dict[str, Any]) -> str:
        status = entry.get("status") or {}
        messages = []
        for item in status.get("messages") or []:
            if len(item) > 1 and isinstance(item[1], (list, tuple)):
                messages.append(str(item[1])[:200])
        return f"ComfyUI 任务执行失败：{'；'.join(messages) or '未知错误'}"

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

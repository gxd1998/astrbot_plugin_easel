# -*- coding: utf-8 -*-
"""OpenAI-compatible images backend (gpt-image-1 and any /images/generations gateway).

Request/Response shape follows the OpenAI Images API; b64_json is preferred,
url is downloaded as a fallback. Model-specific quirks (gpt-image-1 ignores
response_format and always returns b64_json) are tolerated.
"""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import aiohttp

from ..models import MediaRequest, MediaResult
from ..utils import EaselError, humanize_error, monotonic, output_stem


class OpenAIImagesBackend:
    name = "openai_images"
    kind = "image"

    def __init__(self, config: dict[str, Any], out_dir: Path) -> None:
        self._cfg = config or {}
        self._out_dir = out_dir
        self._session: aiohttp.ClientSession | None = None

    def _base(self) -> str:
        base = str(self._cfg.get("base_url") or "https://api.openai.com/v1").strip().rstrip("/")
        return base

    async def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=int(self._cfg.get("timeout_seconds") or 300)))
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def health(self) -> dict:
        key = str(self._cfg.get("api_key") or "")
        if not key:
            return {"ok": False, "error": "openai_images.api_key 未配置"}
        http = await self._http()
        try:
            async with http.get(f"{self._base()}/models",
                                headers={"Authorization": f"Bearer {key}"}) as resp:
                if resp.status != 200:
                    return {"ok": False, "error": f"HTTP {resp.status}"}
                return {"ok": True, "model": self._cfg.get("model")}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": humanize_error(exc, self.name)}

    async def generate(self, request: MediaRequest) -> MediaResult:
        started = monotonic()
        try:
            files = await self._run(request)
            return MediaResult(ok=True, files=files, backend=self.name,
                               workflow=str(self._cfg.get("model") or ""),
                               seconds=monotonic() - started,
                               width=request.width or 0, height=request.height or 0)
        except EaselError as exc:
            return MediaResult(ok=False, backend=self.name, seconds=monotonic() - started, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            return MediaResult(ok=False, backend=self.name,
                               seconds=monotonic() - started, error=humanize_error(exc, self.name))

    async def _run(self, request: MediaRequest) -> list[Path]:
        model = str(self._cfg.get("model") or "gpt-image-1").strip()
        key = str(self._cfg.get("api_key") or "").strip()
        if not key:
            raise EaselError("openai_images.api_key 未配置")
        width = request.width or 1024
        height = request.height or 1024
        if request.images:
            raise EaselError("openai_images 后端暂不支持参考图输入（images/edits 接口将后续接入）")
        # OpenAI sizes are restricted to a few canonical shapes; snap to the closest.
        body: dict[str, Any] = {"model": model, "prompt": request.prompt, "n": 1,
                                "size": self._snap_openai_size(width, height)}
        if model != "gpt-image-1":
            body["response_format"] = "b64_json"

        http = await self._http()
        async with http.post(f"{self._base()}/images/generations",
                             headers={"Authorization": f"Bearer {key}"}, json=body) as resp:
            payload = await resp.json(content_type=None)
            if resp.status != 200:
                err = (payload or {}).get("error") or {}
                raise EaselError(f"OpenAI 错误（HTTP {resp.status} {err.get('code', '')}）：{str(err.get('message'))[:240]}")
        files: list[Path] = []
        for index, item in enumerate((payload or {}).get("data") or []):
            b64 = item.get("b64_json")
            url = str(item.get("url") or "")
            if b64:
                blob = base64.b64decode(b64)
            elif url:
                async with http.get(url) as dl:
                    if dl.status != 200:
                        raise EaselError(f"下载生成图失败：HTTP {dl.status}")
                    blob = await dl.read()
            else:
                raise EaselError("OpenAI 返回项无 b64_json/url")
            out_path = self._out_dir / f"{output_stem('oai')}_{index}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(blob)
            files.append(out_path)
        if not files:
            raise EaselError("OpenAI 返回空结果")
        return files

    @staticmethod
    def _snap_openai_size(width: int, height: int) -> str:
        allowed = [(1024, 1024), (1024, 1536), (1536, 1024)]
        target_ratio = width / max(height, 1)
        best = min(allowed, key=lambda s: abs(s[0] / s[1] - target_ratio))
        return f"{best[0]}x{best[1]}"

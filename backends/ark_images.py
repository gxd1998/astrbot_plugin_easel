# -*- coding: utf-8 -*-
"""Volcengine Ark (Doubao Seedream) images backend.

OpenAI-shaped `/images/generations` with a Bearer API key. Verified against the
live endpoint: errors come back as {"error": {"code", "message"}}, and models
vary on `sequential_image_generation` support, so the field is never sent by
default.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import aiohttp

from ..models import MediaRequest, MediaResult
from ..utils import EaselError, humanize_error, monotonic, output_stem


class ArkImagesBackend:
    name = "ark_images"
    kind = "image"

    def __init__(self, config: dict[str, Any], out_dir: Path) -> None:
        self._cfg = config or {}
        self._out_dir = out_dir
        self._session: aiohttp.ClientSession | None = None

    # -- transport ---------------------------------------------------------
    def _base(self) -> str:
        base = str(self._cfg.get("base_url") or "https://ark.cn-beijing.volces.com/api/v3").strip().rstrip("/")
        return base

    async def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=int(self._cfg.get("timeout_seconds") or 300))
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # -- health ------------------------------------------------------------
    async def health(self) -> dict:
        """Lightweight key check via the models listing (no generation cost)."""
        key = str(self._cfg.get("api_key") or "")
        if not key:
            return {"ok": False, "error": "ark api_key 未配置"}
        http = await self._http()
        try:
            async with http.get(f"{self._base()}/models?max_tokens_limit=1",
                                headers={"Authorization": f"Bearer {key}"}) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return {"ok": False, "error": f"HTTP {resp.status}: {body[:200]}"}
                data = await resp.json()
            models = [m.get("id") for m in data.get("data") or []]
            images = [m for m in models if m and "seedream" in str(m)]
            return {"ok": True, "image_models": images[:6], "model": self._cfg.get("model")}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": humanize_error(exc, self.name)}

    # -- generation --------------------------------------------------------
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
        model = str(self._cfg.get("model") or "").strip()
        if not model:
            raise EaselError("ark_images.model 未配置（控制台模型 ID，如 doubao-seedream-5-0-flash-260915）")
        key = str(self._cfg.get("api_key") or "").strip()
        if not key:
            raise EaselError("ark_images.api_key 未配置")
        width = request.width or 1024
        height = request.height or 1024
        if request.images:
            raise EaselError("ark_images 后端暂不支持参考图输入（Seedream 图生图接口将后续接入）")
        body: dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "size": f"{width}x{height}",
            "response_format": "url",
            "watermark": bool(self._cfg.get("watermark", False)),
        }
        if request.negative:
            # Seedream honours negative instructions inside the prompt tail.
            body["prompt"] = f"{request.prompt}。避免出现：{request.negative}"

        http = await self._http()
        async with http.post(f"{self._base()}/images/generations",
                             headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                             json=body) as resp:
            payload = await resp.json(content_type=None)
            if resp.status != 200:
                raise EaselError(self._translate_error(resp.status, payload))
        items = (payload or {}).get("data") or []
        if not items:
            raise EaselError(f"Ark 返回空结果：{str(payload)[:200]}")

        files: list[Path] = []
        for index, item in enumerate(items):
            url = str(item.get("url") or "")
            b64 = item.get("b64_json")
            blob = await self._fetch_media(url, b64, key)
            out_path = self._out_dir / f"{output_stem('ark')}_{index}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(blob)
            files.append(out_path)
        return files

    async def _fetch_media(self, url: str, b64: str | None, key: str) -> bytes:
        import base64

        if b64:
            return base64.b64decode(b64)
        if not url:
            raise EaselError("Ark 返回项既无 url 也无 b64_json")
        http = await self._http()
        async with http.get(url, headers={"Authorization": f"Bearer {key}"}) as resp:
            if resp.status != 200:
                raise EaselError(f"下载生成图失败：HTTP {resp.status}")
            return await resp.read()

    @staticmethod
    def _translate_error(status: int, payload: Any) -> str:
        err = (payload or {}).get("error") or {}
        code = str(err.get("code") or "")
        message = str(err.get("message") or "")[:260]
        mapping = {
            "ModelNotOpen": f"模型未开通：请在方舟控制台开通该模型（{message}）",
            "SetLimitExceeded": "触发「安全体验模式」限额：请在方舟模型开通页关闭该模式或调整额度",
            "InvalidEndpointOrModel.NotFound": "模型 ID 不存在或无权限：请核对控制台的确切模型 ID",
            "Unauthorized": "鉴权失败：请检查 ark api_key",
        }
        return mapping.get(code, f"Ark 错误（HTTP {status} {code}）：{message}")

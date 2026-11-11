# -*- coding: utf-8 -*-
"""Custom HTTP backend: a JSON request template plus a response path.

Covers any relay/gateway whose API is a single POST returning b64_json or a
url. Template placeholders: {{prompt}}, {{negative}}, {{width}}, {{height}}.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import aiohttp

from ..models import MediaRequest, MediaResult
from ..utils import EaselError, humanize_error, monotonic, output_stem


def render_template(template: Any, request: MediaRequest) -> Any:
    """Recursively substitute placeholders inside a JSON-ish template."""
    if isinstance(template, str):
        out = template
        out = out.replace("{{negative}}", request.negative)
        out = out.replace("{{prompt}}", request.prompt)
        out = out.replace("{{width}}", str(request.width or 1024))
        out = out.replace("{{height}}", str(request.height or 1024))
        return out
    if isinstance(template, dict):
        return {k: render_template(v, request) for k, v in template.items()}
    if isinstance(template, list):
        return [render_template(v, request) for v in template]
    return template


def extract_path(payload: Any, dotted: str) -> Any:
    current = payload
    for part in dotted.split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                raise EaselError(f"response_path 段「{part}」无法索引列表")
        elif isinstance(current, dict):
            if part not in current:
                raise EaselError(f"response_path 段「{part}」不存在于响应")
            current = current[part]
        else:
            raise EaselError(f"response_path 段「{part}」无法继续下钻")
    return current


class CustomHTTPBackend:
    name = "custom"
    kind = "image"

    def __init__(self, name: str, config: dict[str, Any], out_dir: Path) -> None:
        self.name = f"custom:{name}"
        self._key = name
        self._cfg = config or {}
        self._out_dir = out_dir
        self._session: aiohttp.ClientSession | None = None

    async def _http(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=int(self._cfg.get("timeout_seconds") or 300)))
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def health(self) -> dict:
        ok = all([self._cfg.get("url"), self._cfg.get("body_template"), self._cfg.get("response_path")])
        return {"ok": bool(ok), "error": "" if ok else "缺少 url / body_template / response_path 配置"}

    async def generate(self, request: MediaRequest) -> MediaResult:
        started = monotonic()
        try:
            files = await self._run(request)
            return MediaResult(ok=True, files=files, backend=self.name,
                               workflow=self._key, seconds=monotonic() - started,
                               width=request.width or 0, height=request.height or 0)
        except EaselError as exc:
            return MediaResult(ok=False, backend=self.name, seconds=monotonic() - started, error=str(exc))
        except Exception as exc:  # noqa: BLE001
            return MediaResult(ok=False, backend=self.name,
                               seconds=monotonic() - started, error=humanize_error(exc, self.name))

    async def _run(self, request: MediaRequest) -> list[Path]:
        url = str(self._cfg.get("url") or "").strip()
        template = self._cfg.get("body_template")
        response_path = str(self._cfg.get("response_path") or "").strip()
        if not (url and template is not None and response_path):
            raise EaselError("custom 后端需要 url / body_template / response_path 三项配置")
        if request.images:
            raise EaselError("custom 后端暂不支持参考图输入")
        headers = self._cfg.get("headers") if isinstance(self._cfg.get("headers"), dict) else {}
        body = render_template(template, request)
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)

        http = await self._http()
        method = str(self._cfg.get("method") or "POST").upper()
        async with http.request(method, url, data=body if isinstance(body, str) else None,
                                json=body if not isinstance(body, str) else None,
                                headers=headers) as resp:
            text = await resp.text()
            if resp.status != 200:
                raise EaselError(f"custom 后端 HTTP {resp.status}：{text[:240]}")
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                raise EaselError(f"custom 后端返回非 JSON：{text[:240]}")
        found = extract_path(payload, response_path)
        files: list[Path] = []
        items = found if isinstance(found, list) else [found]
        for index, item in enumerate(items):
            blob = await self._resolve_media(str(item), request)
            out_path = self._out_dir / f"{output_stem(self._key)}_{index}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(blob)
            files.append(out_path)
        if not files:
            raise EaselError("custom 后端未取到图片数据")
        return files

    async def _resolve_media(self, value: str, request: MediaRequest) -> bytes:
        if value.startswith("data:image") and "base64," in value:
            return base64.b64decode(value.split("base64,", 1)[1])
        if value.startswith("http://") or value.startswith("https://"):
            http = await self._http()
            async with http.get(value) as resp:
                if resp.status != 200:
                    raise EaselError(f"下载生成图失败：HTTP {resp.status}")
                return await resp.read()
        # bare base64 payload
        try:
            return base64.b64decode(value)
        except Exception as exc:  # noqa: BLE001
            raise EaselError(f"无法解析响应图片数据（前 60 字符）：{value[:60]}") from exc

# -*- coding: utf-8 -*-
"""Small pure helpers shared across layers."""
from __future__ import annotations

import re
import time
import uuid
from datetime import datetime
from typing import Any

_SIZE_RE = re.compile(r"^(\d{2,5})\s*[xX×]\s*(\d{2,5})$")
_RATIO_RE = re.compile(r"^(\d{1,2})\s*[:：]\s*(\d{1,2})$")


class EaselError(ValueError):
    """User-facing generation error; messages are already human readable."""


def parse_size(
    value: str,
    presets: dict[str, Any],
    *,
    default: str = "1:1",
) -> tuple[int, int]:
    """Resolve a size preset name, ratio or explicit WxH into pixels."""
    raw = str(value or default).strip()
    presets = presets or {}
    if raw in presets:
        raw = str(presets[raw])
    match = _SIZE_RE.match(raw)
    if match:
        width, height = int(match.group(1)), int(match.group(2))
    else:
        match = _RATIO_RE.match(raw)
        if not match:
            raise EaselError(f"无法识别画幅「{value}」，可用预设：{', '.join(presets) or '1:1'} 或直接填 1024x1024")
        rw, rh = int(match.group(1)), int(match.group(2))
        if rw == rh:
            width, height = 1024, 1024
        else:
            # Anchor ratios around ~1 MP (long edge 1152) — sane for SD-class models.
            if rw > rh:
                width, height = 1152, 1152 * rh // rw
            else:
                width, height = 1152 * rw // rh, 1152
            width, height = _snap16(width), _snap16(height)
    if not (64 <= width <= 4096 and 64 <= height <= 4096):
        raise EaselError(f"画幅尺寸超界：{width}x{height}（支持 64~4096）")
    return width, height


def _snap16(value: int) -> int:
    return max(64, int(round(value / 16)) * 16)


def random_seed() -> int:
    """ComfyUI accepts plain ints; keep inside the signed 32-bit range."""
    return uuid.uuid4().int % (2**31 - 1)


def output_stem(prefix: str = "easel") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"


def monotonic() -> float:
    return time.monotonic()


def humanize_error(exc: BaseException, backend: str) -> str:
    """Map low-level transport failures to readable Chinese hints."""
    text = str(exc)
    lower = text.lower()
    if "clientconnectorerror" in lower or "connection refused" in lower or "cannot connect" in lower:
        return f"后端 {backend} 连接不上：请确认服务已启动且 base_url 正确"
    if "timeout" in lower:
        return f"后端 {backend} 等待超时，可调大 timeout_seconds 后重试"
    if "404" in text:
        return f"后端 {backend} 返回 404：请检查 base_url 与 API 路径"
    if "401" in text or "403" in text:
        return f"后端 {backend} 鉴权失败（HTTP {text[:3]}）：请检查 api_key"
    return f"后端 {backend} 生成失败：{text[:300]}"

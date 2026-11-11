# -*- coding: utf-8 -*-
"""Backend abstraction shared by ComfyUI, OpenAI-compatible and custom HTTP backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from ..models import MediaRequest, MediaResult


class MediaBackend(ABC):
    """A generation backend. One instance per configured provider."""

    name: ClassVar[str] = "base"
    kind: ClassVar[str] = "image"  # image | video (reserved)

    @abstractmethod
    async def generate(self, request: MediaRequest) -> MediaResult:
        """Run one generation and return a MediaResult; never raise to the caller."""

    @abstractmethod
    async def health(self) -> dict:
        """Return a small diagnostic payload used by /画图测试."""

    async def close(self) -> None:
        """Release any persistent resources; optional."""

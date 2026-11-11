# -*- coding: utf-8 -*-
"""Core data structures shared by every Easel layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MediaRequest:
    """One normalized generation request, backend-agnostic."""

    prompt: str
    negative: str = ""
    size: str = "1:1"
    width: int | None = None
    height: int | None = None
    images: list[Path] = field(default_factory=list)
    workflow_name: str = ""
    backend: str = ""
    seed: int | None = None
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MediaResult:
    """Outcome of a single generation, ready to be delivered or relayed to the LLM."""

    ok: bool
    files: list[Path] = field(default_factory=list)
    backend: str = ""
    workflow: str = ""
    seconds: float = 0.0
    width: int = 0
    height: int = 0
    error: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "backend": self.backend,
            "workflow": self.workflow,
            "seconds": round(self.seconds, 1),
        }
        if self.ok:
            payload["files"] = [str(p) for p in self.files]
            payload["size"] = f"{self.width}x{self.height}" if self.width else ""
            payload["note"] = (
                "Use the built-in send_message_to_user tool with "
                "messages=[{\"type\":\"image\",\"path\":\"<path>\"}] to deliver, "
                "or describe the result in text if delivery is not appropriate."
            )
        else:
            payload["error"] = self.error
        return payload


@dataclass
class WorkflowSlot:
    """A single injectable field inside a workflow graph."""

    node: str
    field: str
    kind: str = "text"  # text | image | video
    feed_mode: str = "auto"  # auto | fixed (image slots only)
    fixed_name: str = ""


@dataclass
class WorkflowDef:
    """A registered workflow: API-format graph plus its slot mapping."""

    name: str
    workflow: dict[str, Any]
    description: str = ""
    slots: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    latent_node: str = ""
    sampler_node: str = ""
    seed_mode: str = "random"  # random | fixed
    default_size: str = "1:1"
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    lora: dict[str, Any] | None = None
    source: str = "builtin"
    fingerprint: str = ""

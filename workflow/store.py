# -*- coding: utf-8 -*-
"""Workflow registry: builtin + user JSON files, fingerprint cache, hot reload."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from ..models import WorkflowDef
from . import mapping


def _warn(message: str) -> None:
    """Log without importing the AstrBot runtime (keeps offline tests working)."""
    try:
        from astrbot.api import logger

        logger.warning(message)
    except Exception:  # noqa: BLE001 - no runtime available, stay silent
        pass


class WorkflowStore:
    """Owns every WorkflowDef; files change -> fingerprint changes -> auto reload."""

    def __init__(self, builtin_dir: Path, user_dir: Path) -> None:
        self._builtin_dir = Path(builtin_dir)
        self._user_dir = Path(user_dir)
        self._overrides: dict[str, Any] = {}
        self._definitions: dict[str, WorkflowDef] = {}
        self.reload()

    # -- config overrides --------------------------------------------------
    def set_overrides(self, overrides: dict[str, Any]) -> None:
        """Apply the `workflows` section of the plugin config on top of files.

        Each entry maps a registry name to {"file": ..., "description": ...,
        "slots": ..., "lora": {...}}; missing fields keep the file's values.
        """
        self._overrides = overrides or {}
        self.reload()

    # -- registry ----------------------------------------------------------
    def reload(self) -> None:
        definitions: dict[str, WorkflowDef] = {}
        for source, root in (("builtin", self._builtin_dir), ("user", self._user_dir)):
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*.json")):
                if path.name.endswith(".meta.json"):
                    continue
                definition = self._load_one(source, path)
                if definition is not None:
                    definitions[definition.name] = definition
        definitions.update(self._apply_overrides(definitions))
        self._definitions = definitions

    def _apply_overrides(self, base: dict[str, WorkflowDef]) -> dict[str, WorkflowDef]:
        merged: dict[str, WorkflowDef] = {}
        for name, entry in (self._overrides or {}).items():
            if not isinstance(entry, dict):
                continue
            definition: WorkflowDef | None = None
            file_name = str(entry.get("file") or "")
            if file_name:
                path = self._user_dir / file_name
                definition = self._load_one("user", path) if path.is_file() else None
                if definition is None:
                    _warn(f"[Easel] 工作流覆盖项 {name} 的文件不可加载：{path}")
                    continue
            else:
                definition = base.get(name)
                if definition is None:
                    _warn(f"[Easel] 工作流覆盖项 {name} 没有对应文件或内置工作流")
                    continue
            definition = copy.deepcopy(definition)
            definition.name = str(name)
            if entry.get("description"):
                definition.description = str(entry["description"])
            if isinstance(entry.get("slots"), dict):
                definition.slots = entry["slots"]
            if isinstance(entry.get("lora"), dict):
                combined = dict(definition.lora or {})
                combined.update(entry["lora"])
                definition.lora = combined
            merged[name] = definition
        return merged

    def _load_one(self, source: str, path: Path) -> WorkflowDef | None:
        try:
            workflow = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(workflow, dict) or not workflow:
            return None
        meta_path = path.with_suffix(".meta.json")
        meta: dict[str, Any] = {}
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
            except (OSError, json.JSONDecodeError):
                meta = {}
        name = str(meta.get("name") or path.stem)
        slots = meta.get("slots") if isinstance(meta.get("slots"), dict) else mapping.infer(workflow)
        fingerprint = hashlib.sha256(
            json.dumps(workflow, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:12]
        return WorkflowDef(
            name=name,
            workflow=workflow,
            description=str(meta.get("description") or ""),
            slots=slots,
            latent_node=str(meta.get("latent_node") or ""),
            sampler_node=str(meta.get("sampler_node") or ""),
            seed_mode=str(meta.get("seed_mode") or "random"),
            default_size=str(meta.get("default_size") or "1:1"),
            lora=meta.get("lora") if isinstance(meta.get("lora"), dict) else None,
            source=source,
            fingerprint=fingerprint,
        )

    # -- access ------------------------------------------------------------
    def get(self, name: str) -> WorkflowDef | None:
        return self._definitions.get(str(name or ""))

    def names(self) -> list[str]:
        return sorted(self._definitions)

    def summaries(self) -> list[dict[str, Any]]:
        rows = []
        for name in self.names():
            definition = self._definitions[name]
            problems = mapping.validate(definition.workflow, {"slots": definition.slots})
            rows.append({
                "name": name,
                "source": definition.source,
                "description": definition.description,
                "fingerprint": definition.fingerprint,
                "slots": definition.slots,
                "problems": problems,
            })
        return rows

    def default(self) -> WorkflowDef | None:
        for candidate in ("qwen_t2i",):
            definition = self._definitions.get(candidate)
            if definition:
                return definition
        return next(iter(self._definitions.values()), None)

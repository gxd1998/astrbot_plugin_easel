# -*- coding: utf-8 -*-
"""Workflow registry and slot mapping."""
from .mapping import apply_request, infer, inject_lora, validate
from .store import WorkflowStore

__all__ = ["WorkflowStore", "infer", "validate", "apply_request", "inject_lora"]

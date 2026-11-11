# -*- coding: utf-8 -*-
"""Generation backends."""
from .base import MediaBackend
from .comfyui import ComfyUIBackend

__all__ = ["MediaBackend", "ComfyUIBackend"]

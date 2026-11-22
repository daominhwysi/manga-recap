"""Registry for video renderers."""

from src.core.protocols import VideoRenderer
from src.core.registry import Registry

RENDERER_REGISTRY: Registry[VideoRenderer] = Registry("video_renderer")

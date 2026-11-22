"""Registry for timeline exporters."""

from src.core.protocols import Exporter
from src.core.registry import Registry

EXPORTER_REGISTRY: Registry[Exporter] = Registry("exporter")

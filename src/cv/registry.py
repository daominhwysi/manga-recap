"""Registry for CV panel detectors."""

from src.core.protocols import PanelDetector
from src.core.registry import Registry

DETECTOR_REGISTRY: Registry[PanelDetector] = Registry("panel_detector")

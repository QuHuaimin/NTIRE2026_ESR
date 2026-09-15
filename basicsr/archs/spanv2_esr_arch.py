"""BasicSR registration wrapper for the official NTIRE Team 22 model."""

import importlib.util
from pathlib import Path

from basicsr.utils.registry import ARCH_REGISTRY


_OFFICIAL_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / 'models' / 'team22_SPANV2_ESR.py')
_SPEC = importlib.util.spec_from_file_location(
    'spanv2_official_team22_model', _OFFICIAL_MODEL_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
OfficialSPANV2ESR = _MODULE.SPANV2_ESR


@ARCH_REGISTRY.register()
class SPANV2ESR(OfficialSPANV2ESR):
    """Expose the unmodified official model through BasicSR's registry."""

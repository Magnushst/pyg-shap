# src/pyg_captum_shap/__init__.py
from .wrapper import PyGCaptumWrapper
from .explainer import compute_shap_values

__version__ = "0.1.3"
__all__ = ["PyGCaptumWrapper", "compute_shap_values"]

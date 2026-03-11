# src/pyg_captum_shap/__init__.py
from .wrapper import PyGCaptumWrapper
from .explainer import compute_gat_shap_values

__version__ = "0.1.2"
__all__ = ["PyGCaptumWrapper", "compute_gat_shap_values"]

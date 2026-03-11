# src/pyg_captum_shap/explainer.py
import torch
from captum.attr import GradientShap
from .wrapper import PyGCaptumWrapper

def compute_shap_values(model, target_graph, target_task: int = 0, n_samples: int = 5):
    """
    Extracts node-level SHAP values for a specified graph and task.
    """
    model.eval()
    device = next(model.parameters()).device
    target_graph = target_graph.to(device)

    with torch.no_grad():
        target_embedded = model.node_emb(target_graph.x).unsqueeze(0)
        baselines_embedded = torch.zeros_like(target_embedded)

    g_feat = getattr(target_graph, 'global_features', None)
    if g_feat is not None and g_feat.dim() == 1:
        g_feat = g_feat.unsqueeze(0)

    graph_dict = {
        'edge_index': target_graph.edge_index,
        'edge_attr': getattr(target_graph, 'edge_attr', None),
        'global_features': g_feat
    }

    wrapper = PyGCaptumWrapper(model)
    explainer = GradientShap(wrapper)

    attributions, delta = explainer.attribute(
        inputs=target_embedded,
        baselines=baselines_embedded,
        additional_forward_args=(graph_dict,),
        target=target_task,
        n_samples=n_samples, 
        return_convergence_delta=True
    )

    node_attributions = attributions.squeeze(0).sum(dim=1)
    return node_attributions.cpu()

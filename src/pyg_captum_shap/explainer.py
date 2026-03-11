# src/pyg_captum_shap/explainer.py
import torch
from captum.attr import GradientShap
from .wrapper import PyGCaptumWrapper

def compute_shap_values(model, target_graph, target_task: int = 0, n_samples: int = 25):
    """
    Extracts Node, Edge (optional), and Global (optional) SHAP values.
    Returns a dictionary for clarity.
    """
    model.eval()
    device = next(model.parameters()).device
    target_graph = target_graph.to(device)

    input_list = []
    baseline_list = []
    input_types = []

    # Nodes (Mandatory)
    node_emb = model.node_emb(target_graph.x).unsqueeze(0)
    input_list.append(node_emb)
    baseline_list.append(torch.zeros_like(node_emb))
    input_types.append('nodes')

    # Edges (Optional)
    edge_attr = getattr(target_graph, 'edge_attr', None)
    if edge_attr is not None:
        edge_attr_t = edge_attr.unsqueeze(0)
        input_list.append(edge_attr_t)
        baseline_list.append(torch.zeros_like(edge_attr_t))
        input_types.append('edges')

    # Global (Optional)
    g_feat = getattr(target_graph, 'global_features', None)
    if g_feat is not None:
        g_feat_t = g_feat.unsqueeze(0) if g_feat.dim() == 1 else g_feat
        input_list.append(g_feat_t)
        baseline_list.append(torch.zeros_like(g_feat_t))
        input_types.append('global')

    # Setup Wrapper & Explainer
    graph_dict = {'edge_index': target_graph.edge_index}
    wrapper = PyGCaptumWrapper(model)
    explainer = GradientShap(wrapper)

    # Ppass None for edge_inputs or global_inputs in additional_args 
    # if they weren't in the main inputs tuple.
    attr_tuple = explainer.attribute(
        inputs=tuple(input_list),
        baselines=tuple(baseline_list),
        additional_forward_args=(graph_dict,),
        target=target_task,
        n_samples=n_samples
    )

    # Map results to a dictionary
    results = {}
    for i, name in enumerate(input_types):
        results[name] = attr_tuple[i].squeeze(0).cpu()

    return results
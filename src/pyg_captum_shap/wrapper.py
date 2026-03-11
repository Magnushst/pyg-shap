# src/pyg_captum_shap/wrapper.py
import torch

class PyGCaptumWrapper(torch.nn.Module):
    """
    A wrapper to shield PyTorch Geometric structural tensors from Captum's
    Euclidean sampling mechanisms.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs, graph_dict):
        B, N, F = inputs.shape
        x = inputs.view(-1, F) 

        edge_index = graph_dict['edge_index']
        edge_attr = graph_dict.get('edge_attr')
        global_features = graph_dict.get('global_features')

        batch = torch.arange(B, device=inputs.device).repeat_interleave(N)

        edge_indices = []
        for i in range(B):
            edge_indices.append(edge_index + i * N)
        batched_edge_index = torch.cat(edge_indices, dim=1) if B > 0 else edge_index

        if edge_attr is not None:
            batched_edge_attr = edge_attr.repeat(B, 1) if edge_attr.dim() > 1 else edge_attr.repeat(B)
        else:
            batched_edge_attr = None

        if global_features is not None:
            batched_global = global_features.repeat(B, 1) if global_features.dim() > 1 else global_features.repeat(B)
        else:
            batched_global = None

        return self.model(
            x=x, 
            edge_index=batched_edge_index, 
            batch=batch, 
            edge_attr=batched_edge_attr, 
            global_features=batched_global, 
            apply_embedding=False
        )

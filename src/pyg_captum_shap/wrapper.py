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

    def forward(self, node_inputs, edge_inputs, global_inputs, graph_dict):
        B, N, F = node_inputs.shape
        x = node_inputs.view(-1, F)
        
        # Handle edge attributes 
        batched_edge_attr = None
        if edge_inputs is not None:
            # Flatten [B, E, Ef] -> [B*E, Ef]
            batched_edge_attr = edge_inputs.view(-1, edge_inputs.shape[-1])
        
        # Reconstruct block-diagonal edge_index
        edge_index = graph_dict['edge_index']
        # Pre-calculate indices for efficiency
        offsets = torch.arange(B, device=node_inputs.device).view(-1, 1, 1) * N
        batched_edge_index = (edge_index.unsqueeze(0) + offsets).transpose(0, 1).reshape(2, -1)

        # Create batch vector
        batch_vec = torch.arange(B, device=node_inputs.device).repeat_interleave(N)

        # Global Features are already [B, G], no flattening needed
        # Pass them as-is. If they are None, the model handles it.
        return self.model(
            x=x, 
            edge_index=batched_edge_index, 
            batch=batch_vec, 
            edge_attr=batched_edge_attr, 
            global_features=global_inputs, 
            apply_embedding=False
        )

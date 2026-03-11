# PyG-Captum-sSHAP

A robust wrapper bridging the Euclidean sampling mechanisms of Captum with the non-Euclidean batching of PyTorch Geometric (PyG). Designed specifically for molecular QSAR modelling.
This library is built on top of the Captum library by PyTorch.

This library provides a **Dictionary-Shielded Wrapper** that:
1.  Intercepts Captum's corrupted sampling tensors.
2.  Safely reconstructs the block-diagonal graph batches on-the-fly.
3.  Allows for seamless SHAP and Integrated Gradients attribution on complex PyG architectures (GAT, GCN, Transformers).

## Installation
```bash
pip install pyg-captum-shaps
```

## Quick Start
```python
from pyg_captum_shap import compute_gat_shap_values

# Extract SHAP values for a specific molecule and task
node_attributions = compute_gat_shap_values(
    model=your_trained_model,
    target_graph=molecule_graph_data,
    target_task=0
)

# node_attributions now contains the importance score for every atom in the graph
```

## License
Distributed under the MIT License.

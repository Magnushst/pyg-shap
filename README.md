[![PyPI version](https://img.shields.io/pypi/v/pyg-captum-shap.svg)](https://pypi.org/project/pyg-captum-shap/)

# PyG-Captum-SHAP

A robust wrapper bridging the Euclidean sampling mechanisms of **Captum** with the non-Euclidean batching of **PyTorch Geometric (PyG)**. 

Designed specifically for molecular QSAR modelling, this library solves the dimension mismatch errors (e.g., `Expected size 2, got 10`) that occur when using Captum on Graph Neural Networks.

## Key Features
1.  **Dictionary-Shielded Wrapper**: Protects structural tensors (`edge_index`) from corruption during Captum sampling.
2.  **Multi-Input Support**: Generate attributions for **Nodes (Atoms)**, **Edges (Bonds)**, and **Global Molecular Features** simultaneously.
3.  **Automatic Reconstruction**: On-the-fly reconstruction of block-diagonal graph batches for Captum's internal forward passes.

## Installation
```bash
pip install pyg-captum-shap
```

## Quick Start
```python
from pyg_captum_shap import compute_shap_values

# Extract attributions for a specific molecule and task
results = compute_shap_values(
    model=your_trained_model,
    target_graph=molecule_graph_data,
    target_task=0,
    n_samples=25
)

# Access node, edge, and global SHAP values
node_importance = results['nodes']          # Shape: [N, F]
edge_importance = results.get('edges')      # Optional
global_importance = results.get('global')   # Optional

# node_attributions now contains the importance score for every atom in the graph
```

## License
Distributed under the MIT License. Built on top of the Captum library by PyTorch.
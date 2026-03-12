[![PyPI version](https://img.shields.io/pypi/v/pyg-captum-shap.svg)](https://pypi.org/project/pyg-captum-shap/)

# PyG-Captum-SHAP

A robust wrapper bridging the Euclidean sampling mechanisms of **Captum** with the non-Euclidean batching of **PyTorch Geometric (PyG)**. 

Designed specifically for complex molecular QSAR modelling, this library resolves the dimensional mismatch errors (e.g., `Expected size 2, got 10`) that occur when applying Captum to Graph Neural Networks, whilst enabling simultaneous attribution across multiple graph modalities.

## Beyond Native Explainability: The Multi-Modal Bottleneck

While PyTorch Geometric provides native explainability utilities such as `to_captum_model` and the `torch_geometric.explain` module, these tools are architecturally constrained to simple spatial topologies. 

Native PyG explainers rely on hardcoded mask types (`mask_type='node'` or `'edge'`) and automatically relegate any auxiliary graph-level tensors to `additional_forward_args`. Because Captum strictly computes gradients only for tensors passed within its primary `inputs` tuple, native PyG utilities render high-dimensional global descriptors (e.g., MolFormer embeddings, RDKit descriptors, or topological signatures) mathematically invisible to the attribution algorithm.

**PyG-Captum-SHAP** resolves this fundamental limitation. It dynamically packs nodes, edges, and global features into the primary attribution tuple, whilst exclusively shielding the `edge_index` dictionary. This bypasses the native PyG constraints, enabling true multi-modal SHAP extraction across advanced neural architectures.

## Key Features
1.  **Dictionary-Shielded Wrapper**: Protects structural tensors (`edge_index`) from corruption during Captum's internal feature perturbation and sampling phases.
2.  **Multi-Input Support**: Generates mathematically consistent attributions for **Nodes (Atoms)**, **Edges (Bonds)**, and **Global Molecular Features** simultaneously.
3.  **Automatic Reconstruction**: Performs on-the-fly reconstruction of block-diagonal graph batches for Captum's internal forward passes.

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
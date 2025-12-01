# 1D vs 2D Lookup KAN Comparison Example

This Jupyter notebook provides a comprehensive comparison between 1D and 2D lookup-based Kolmogorov-Arnold Networks (KANs) for function approximation.

## Overview

The notebook replicates the paper's benchmark: approximating a randomly-initialized MLP (R^32 → R^1) using lookup-based KAN architectures.

## Quick Start

```bash
# From the KAN directory
cd examples
jupyter notebook example_1d_vs_2d_comparison.ipynb
```

## Requirements

### Minimum (1D KAN only)
- PyTorch (CPU or CUDA)
- NumPy
- Matplotlib

The notebook will work with **1D KAN only** (pure PyTorch implementation) without any compilation.

### Full Experience (1D + 2D KAN)
To use the 2D KAN (lmKAN) with CUDA acceleration:

1. **CUDA Toolkit** with `nvcc` compiler
2. **Install lmKAN**:
   ```bash
   cd ../lmkan
   export CUDA_HOME=/path/to/cuda  # Set your CUDA path
   pip install .
   ```

**Note**: If you don't have CUDA/nvcc, the notebook will gracefully skip 2D KAN sections and work with 1D KAN only.

## Contents

### 1. Setup and Verification
- CUDA availability check
- lmKAN package import verification (optional)
- Utility functions for CDF grid generation

### 2. Model Implementations
- **1D Lookup KAN**: Simple PyTorch implementation using linear interpolation on non-uniform CDF grid
- **2D Lookup KAN (lmKAN)**: CUDA-optimized bilinear interpolation on paired inputs (if available)

### 3. Benchmark Task
- Target: Frozen random MLP (32 → 64 → 1 with ReLU)
- Dataset: 10,000 training samples, 2,000 test samples
- Training: 200 epochs with Hessian regularization schedule

### 4. Comprehensive Visualizations
- **CDF Grid**: Non-uniform Laplace CDF-based grid borders
- **1D Basis Functions**: Linear interpolation "hat" functions with partition of unity verification
- **2D Basis Functions**: Bilinear interpolation tensor products (both 3D surfaces and heatmaps) - if 2D KAN available
- **Learned Functions**: Visualization of trained lookup tables from both models

### 5. Architecture Explanation
- Data flow diagrams
- Parameter tensor structures
- Comparison of 1D vs 2D approaches

## Notebook Sections

1. Setup and Imports
2. Utility Functions: CDF Grid
3. Implement 1D Lookup KAN Layer
4. Setup 2D Lookup KAN Layer (lmKAN) - *Optional: skipped if CUDA unavailable*
5. Create Target Function: Random MLP
6. Training Utilities
7. Train and Compare Models
8. Compare Results
9. Visualize CDF Grid
10. Visualize 1D Spline Basis Functions
11. Visualize 2D Spline Basis Functions - *Optional: skipped if 2D KAN unavailable*
12. Visualize Learned Functions
13. Summary and Conclusions
14. Layer Architecture Explanation
15. Export Results

## Testing

A test script is provided to verify the setup:

```bash
python test_example.py
```

This will test:
- ✓ Import checks
- ✓ CDF grid generation
- ✓ 1D KAN implementation
- ✓ 2D KAN (if CUDA available)
- ✓ Target MLP and training
- ✓ Basis function computation
- ✓ Visualization

## Output

The notebook generates:
- Training curves comparing 1D vs 2D KAN (or 1D only)
- Hessian regularization schedule visualization
- CDF grid plots
- 1D and 2D basis function visualizations (multiple representations)
- Learned function heatmaps
- `comparison_results.json` with final metrics

## Key Results (Expected)

### With 1D KAN Only
- **1D KAN**: ~288 parameters, moderate approximation quality
- Clear visualization of linear interpolation basis functions
- Complete understanding of 1D lookup KAN architecture

### With Both 1D and 2D KAN
- **1D KAN**: ~288 parameters, moderate approximation quality
- **2D KAN**: ~1,296 parameters (~4.5x more), significantly better MSE
- Direct comparison showing advantages of 2D approach
- Visualization of bilinear vs linear interpolation

## Files in this Directory

- `example_1d_vs_2d_comparison.ipynb` - Main notebook (35 cells)
- `EXAMPLE_README.md` - This file
- `IMPLEMENTATION_SUMMARY.md` - Technical details
- `test_example.py` - Test script
- `test_grid.png` - Generated test plot (after running test)

## Troubleshooting

### "No module named 'lmKAN_kernels'"
This means lmKAN is not installed. The notebook will work with 1D KAN only. To install lmKAN:
1. Ensure CUDA toolkit is installed
2. Set `CUDA_HOME` environment variable
3. Run `pip install .` in the `lmkan` directory

### "CUDA not available"
The notebook will work on CPU with 1D KAN. For 2D KAN, you need a CUDA-capable GPU.

### Import errors
Ensure you have:
```bash
pip install torch numpy matplotlib jupyter
```

## References

- Paper: "Lookup multivariate Kolmogorov-Arnold Networks" (arXiv:2509.07103)
- See `../lmkan/README.rst` for more details on lmKAN

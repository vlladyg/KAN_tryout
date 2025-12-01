# KAN Examples

This directory contains examples and tutorials for the Lookup Multivariate Kolmogorov-Arnold Networks (lmKAN) implementation.

## Available Examples

### 1. 1D vs 2D Lookup KAN Comparison

**File**: `example_1d_vs_2d_comparison.ipynb`

A comprehensive Jupyter notebook comparing 1D and 2D lookup-based KAN architectures for function approximation.

**Quick Start**:
```bash
jupyter notebook example_1d_vs_2d_comparison.ipynb
```

**What's Included**:
- Implementation of 1D Lookup KAN (pure PyTorch)
- Integration with 2D lmKAN (CUDA-accelerated, optional)
- Training on random MLP benchmark (R^32 → R^1)
- Comprehensive visualizations:
  - CDF grid structure
  - 1D and 2D basis functions
  - Learned lookup tables
- Layer architecture explanations
- Performance comparisons

**Requirements**:
- **Minimum**: PyTorch, NumPy, Matplotlib (works with 1D KAN only)
- **Full**: CUDA + lmKAN installation (for 2D KAN)

See `EXAMPLE_README.md` for detailed instructions.

## Testing

Verify your setup with the test script:

```bash
python test_example.py
```

This will check:
- ✓ All required imports
- ✓ CDF grid generation
- ✓ 1D KAN implementation
- ✓ 2D KAN (if CUDA available)
- ✓ Training functionality
- ✓ Basis function computations
- ✓ Visualization capabilities

## Directory Structure

```
examples/
├── README.md                           # This file
├── example_1d_vs_2d_comparison.ipynb  # Main comparison notebook
├── EXAMPLE_README.md                   # Detailed guide for the notebook
├── IMPLEMENTATION_SUMMARY.md           # Technical implementation details
├── test_example.py                     # Test script
└── test_grid.png                       # Generated test plot
```

## Notes

### Working Without CUDA

The notebook is designed to work **without CUDA compilation**. If lmKAN is not installed or CUDA is unavailable:
- The notebook will use the 1D KAN implementation (pure PyTorch)
- 2D KAN sections will be gracefully skipped
- All visualizations except 2D-specific ones will work
- You'll still learn about lookup KAN architectures and basis functions

### Installing lmKAN (Optional)

For the full experience with 2D KAN:

1. Ensure CUDA toolkit is installed with `nvcc`
2. Set your CUDA path:
   ```bash
   export CUDA_HOME=/usr/local/cuda  # Adjust to your CUDA installation
   ```
3. Install lmKAN:
   ```bash
   cd ../lmkan
   pip install .
   ```

## Getting Help

- **lmKAN Documentation**: See `../lmkan/README.rst`
- **Paper**: arXiv:2509.07103 - "Lookup multivariate Kolmogorov-Arnold Networks"
- **Issues**: Check the notebook cells for inline documentation and explanations

## Output Files

When you run the notebook, it will generate:
- `comparison_results.json` - Final metrics and comparisons
- Various plots showing training curves, basis functions, and learned functions

## Citation

If you use this code or the lmKAN implementation, please cite:

```bibtex
@article{pozdnyakov2025lookup,
  title={Lookup multivariate Kolmogorov-Arnold Networks},
  author={Pozdnyakov, Sergey and Schwaller, Philippe},
  journal={arXiv preprint arXiv:2509.07103},
  year={2025}
}
```


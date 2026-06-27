# SAGD for Robust Inverse Design of Nonlocal Thin Films

This repository provides a PyTorch implementation of **Sharpness-Aware Gradient Descent (SAGD)** for the robust inverse design of nonlocal multilayer thin films.

The code implements a differentiable transfer-matrix-method (TMM) solver and optimizes layer thicknesses to achieve target angle- and wavelength-dependent optical responses.

## Paper

Yuchen Ma, Guangwei Hu and Liangcai Cao.  
**Sharpness-Aware Minimization for Robust Inverse Design of Nonlocal Thin Films.**  
*Laser & Photonics Reviews* (2026).  
DOI: [10.1002/lpor.71366](https://doi.org/10.1002/lpor.71366)

## Features

- Differentiable coherent TMM implemented in PyTorch
- Sharpness-aware optimization for improved fabrication robustness
- GPU-compatible automatic differentiation
- Visualization of convergence and angle–wavelength transmittance maps

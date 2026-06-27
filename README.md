# SAGD for Robust Inverse Design of Nonlocal Thin Films

A differentiable physics solver for designing optical thin-film multilayers. This project uses **Sharpness-Aware Minimization (SAM/SAGD)** to find robust layer thickness configurations that achieve target transmittance spectra.

Powered by PyTorch, it supports GPU acceleration and automatic differentiation through the Transfer Matrix Method (TMM).
<img width="1600" height="1151" alt="image" src="https://github.com/user-attachments/assets/3be39101-0dd0-4f9e-9787-1acbd2404f80" />

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

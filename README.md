# SAGD for Robust Inverse Design of Nonlocal Thin Films

A differentiable physics solver for designing optical thin-film multilayers. This project uses **Sharpness-Aware Minimization (SAM/SAGD)** to find robust layer thickness configurations that achieve target transmittance spectra.

Powered by PyTorch, it supports GPU acceleration and automatic differentiation through the Transfer Matrix Method (TMM).

<div align="center">
  <img width="70%" alt="fig1" src="https://github.com/user-attachments/assets/f03eaabe-ec1c-4631-befd-97a1296d988b" />
</div>

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

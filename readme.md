# Thin-Film Optimization via SAGD (PyTorch)

A differentiable physics solver for designing optical thin-film multilayers. This project uses **Sharpness-Aware Minimization (SAM)** to find robust layer thickness configurations that achieve target transmittance spectra.

Powered by **PyTorch**, it supports GPU acceleration and automatic differentiation through the Transfer Matrix Method (TMM).

## ✨ Features

* **Differentiable TMM**: Fully vectorized Coherent Transfer Matrix Method implemented as a PyTorch Module.
* **Robust Optimization**: Uses SAGD (SGD with SAM) to find flat minima, improving fabrication tolerance.
* **Visualization**: Automatically generates convergence plots and high-resolution transmittance spectrum heatmaps.

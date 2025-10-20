# Search for New Physics with Anomaly Detection at the LHC

Bachelor's thesis by Valerio Tinari - Sapienza University of Rome (2023/2024)

## Overview

This project implements an **unsupervised anomaly detection** approach using Deep Learning to search for new physics beyond the Standard Model at the Large Hadron Collider (LHC). The model uses a **Transformer architecture** to distinguish between Standard Model background events and potential new physics signals in particle collision data.

## Research Problem

Traditional model-dependent searches for new physics (like Dark Matter) have been unsuccessful. This work explores a **model-agnostic approach** that can:
- Detect anomalies without assuming a specific theoretical model
- Focus on micro-structures within particle jets
- Provide hints about where to look more deeply with precision techniques

## Dataset

**LHC Olympics 2020 R&D Dataset:**
- 1 million background events (QCD dijet events)
- 100,000 signal events (Z' → XY resonance)
- Monte Carlo simulated data using Pythia 8.219 and Delphes 3.4.1
- Features: transverse momentum (p_T), pseudorapidity (η), azimuthal angle (φ)

## Methodology

### Data Preprocessing
1. **Jet Clustering:** Anti-k_t algorithm (R=1) to reconstruct jets
2. **Feature Selection:** Up to 50 constituents per jet, 3 features each
3. **Dataset Transformations:**
   - **Standard:** Direct reconstruction without transformation
   - **Rotated:** Two Lorentz boosts to set η'_jet = 0, φ'_jet = 0
   - **Transformed:** Mass rescaling + energy boost + rotation

### Model Architecture

**Transformer Encoder:**
- Multi-Head Attention (MHA) mechanism with dynamic weights
- L stacked layers (optimized during training)
- GELU activation function
- Dropout (p=0.30) for regularization
- Input dimension: (batch_size, 50, 3)
- Embedding dimension: 128

### Training Strategy

**Unsupervised Learning:**
- Training set: 1.6M background events only
- Validation set: 200k background events
- Test set: 200k background + 200k signal events
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam with optimized learning rate

**Anomaly Score:** Computed as L2 norm between input and reconstructed output

## Results

| Dataset | Learning Rate | Heads | Layers | Best Epoch | AUC Score |
|---------|---------------|-------|--------|------------|-----------|
| Standard (anti-k_T) | 10⁻⁵ | 8 | 5 | 10 | **53.0%** ❌ |
| Rotated | 10⁻⁶ | 8 | 32 | 27 | **66.4%** ✓ |
| Transformed | 10⁻⁶ | 2 | 4 | 47 | **69.0%** ✓✓ |

### Key Findings

1. **Standard dataset fails:** Acts as random classifier (AUC ≈ 53%) due to rotational symmetries in data
2. **Rotation helps significantly:** Removing symmetries allows model to focus on jet micro-structures (AUC = 66.4%)
3. **Transformation performs best:** Frame change reduces jet spatial aperture, making classification easier (AUC = 69.0%)

## Physical Interpretation

The **transformed dataset** achieves best results because:
- Lorentz boosts push jets forward in detector
- Lower energy (E₀ = 1 GeV) reduces spatial aperture
- Smaller coverage in (η, φ) plane helps model learn relevant features
- Focus shifts from macro-structure to micro-structure details

## Limitations & Future Work

- AUC of 69% is acceptable but not ideal for detecting small SM deviations
- Hyperparameters may need further optimization
- Alternative architectures could be explored
- Model-agnostic approach trades specificity for broader applicability

## Technical Stack

- **Languages:** Python
- **Deep Learning:** PyTorch
- **Data Processing:** NumPy, Scikit-learn
- **Jet Clustering:** Fastjet (anti-k_t implementation)
- **Simulation:** Pythia 8.219, Delphes 3.4.1
- **Typesetting:** LaTeX with Sapthesis class

## Key Concepts

- **ATLAS Detector:** Cylindrical symmetry detector at LHC interaction point 1
- **Pseudo-rapidity (η):** Related to polar angle θ by η = -ln(tan(θ/2))
- **Anomaly Detection:** Model-agnostic search for outliers/rare events
- **Multi-Head Attention:** Parallel attention mechanisms with dynamic weights
- **ROC Curve:** Trade-off between True Positive Rate and False Positive Rate


## Contact

**Author:** Valerio Tinari  
**Email:** tinari.1998628@studenti.uniroma1.it  
**Institution:** Sapienza University of Rome  
**Advisors:** Prof. Stefano Giagu, Dr. Graziella Russo

---

© 2024 Valerio Tinari. All rights reserved.

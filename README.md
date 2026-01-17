
# Robustness and Calibration Analysis of Fine-Tuned Transformer Models under Distribution Shift

## Overview

Transformer-based language models achieve strong benchmark accuracy, yet their
reliability under real-world input variations remains insufficiently understood.
In production settings, confidence estimates are often as critical as raw accuracy,
as overconfident incorrect predictions can lead to severe downstream failures.

This project investigates the robustness and probabilistic calibration of a
fine-tuned Transformer classifier under prompt-level and token-level input
perturbations, highlighting failure modes not captured by accuracy metrics alone.

---

## Objectives

- Fine-tune a pretrained Transformer model for sentiment classification
- Evaluate robustness under realistic input perturbations
- Analyze confidence behavior and probabilistic calibration
- Quantify reliability degradation using calibration metrics and visualizations

---

## Model and Dataset

- **Model:** DistilBERT (pretrained, fine-tuned end-to-end)
- **Task:** Binary sentiment classification
- **Dataset:** SST-2 (GLUE benchmark, ~67k samples)
- **Framework:** PyTorch, HuggingFace Transformers

DistilBERT was selected to balance representational capacity and computational
efficiency, enabling rapid experimentation while remaining representative of
modern Transformer-based classifiers.

---

## Experimental Setup

### Fine-Tuning
The model is fine-tuned using cross-entropy loss with AdamW optimization.
Training duration is intentionally constrained to avoid overfitting, as excessive
confidence amplification negatively impacts calibration under distribution shift.

### Perturbation Strategies
To simulate realistic deployment scenarios, inference-time perturbations are applied:

- **Prompt perturbations:** Input prefixes such as "Review:" and "User opinion:"
- **Token-level noise:** Random token dropout preserving semantic structure

These perturbations alter surface form while maintaining label semantics.

---

## Evaluation Metrics

In addition to standard accuracy, the following metrics are analyzed:

- Prediction confidence (maximum softmax probability)
- Expected Calibration Error (ECE, 10-bin)
- Reliability diagrams

Calibration metrics quantify the alignment between predicted confidence and
empirical correctness frequency.

---

## Key Results

- Validation accuracy remains high (~90%) on clean inputs
- Robust accuracy degrades moderately under perturbations
- Expected Calibration Error increases from **0.074 to 0.103** (≈38% relative increase)
- Reliability diagrams reveal systematic overconfidence under distribution shift

These findings demonstrate that accuracy alone significantly underestimates
model risk in perturbed settings.

---

## Insights and Implications

The analysis reveals that Transformer classifiers can remain highly confident even
when predictions are incorrect under distribution shift. This confidence–accuracy
misalignment poses risks in safety-critical and decision-making systems.

The results highlight the importance of calibration-aware evaluation for deployed
NLP models.

---

## Limitations

- Analysis limited to binary sentiment classification
- Perturbations are heuristic rather than adversarial
- Calibration improvement methods are not applied

---

## Future Work

- Temperature scaling and post-hoc calibration methods
- Calibration-aware training objectives
- Extension to multi-class and long-form NLP tasks
- Robustness evaluation across multiple Transformer architectures

---



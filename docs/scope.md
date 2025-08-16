# Project Scope – Multimodal Disaster Response AI

## Objective
Build a **proof-of-concept multimodal AI** system that detects flood-related disasters using both **text (tweets)** and **images (drone photos)**.

## Why
- Showcase ability to handle **NLP + Computer Vision + Fusion** in one pipeline.
- Build a **portfolio-quality project** suitable for internship applications (Google STEP/AI residency).

## Deliverables
1. Balanced, preprocessed text and image datasets.
2. Text-only classifier (baseline BERT/DistilBERT).
3. Image-only classifier (baseline ResNet18/MobileNetV2).
4. Fusion model combining text + image features.
5. Streamlit demo app for interactive exploration.
6. Documentation (this scope, datasheet, README).

## Out of Scope
- Real-time disaster monitoring.
- Deployment in emergency contexts.
- Handling non-flood disaster modalities (earthquakes, fires, etc.).
- Large-scale scalability (focus is clarity + reproducibility).

## Assumptions
- Laptop training (≤16GB RAM, no GPU cluster).
- Pretrained models (transfer learning).
- Open-source datasets only.

## Risks & Mitigations
- **Imbalanced data** → solved via balanced sampling.
- **Overfitting** (small data) → mitigated with augmentations & regularization.
- **Bias** (English-only, limited locations) → acknowledged in datasheet.

## Success Criteria
- End-to-end pipeline reproducible with provided scripts.
- Baseline text & image models achieve >70% accuracy on test splits.
- Fusion model shows measurable improvement over single-modality baselines.
- Clear, ethical documentation present.

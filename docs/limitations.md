# Limitations of Multimodal Disaster AI

This document records observed failure cases during robustness and adversarial stress testing.

## 1. Ambiguous Text (Metaphors & Sarcasm)
- **Input:** “That concert was fire! 🔥”
- **Predicted:** non_disaster (p=0.24)
- **Ground Truth:** non_disaster ✅
- **Observation:** Correct in this case, but model confidence shows uncertainty. Sarcasm and metaphorical use of “fire,” “storm,” or “hurricane” could cause misclassifications in other contexts.

## 2. Figurative Language in Music/Entertainment
- **Input:** “The new song ‘Hurricane’ is trending everywhere”
- **Predicted:** non_disaster (p=0.20)
- **Ground Truth:** non_disaster ✅
- **Observation:** Correct here, but if the text had stronger disaster cues, the model might confuse cultural references with real disasters.

## 3. Edge Case Visuals – Swimming Pool
- **Image:** swimming_pool.png
- **Predicted:** non_disaster (p≈0.0)
- **Observation:** Model correctly ignored water in a pool. However, reliance on context (e.g., flooding vs. contained water) is fragile and may fail on unusual angles or lighting.

## 4. Edge Case Visuals – Glass Building
- **Image:** glass_building.png
- **Predicted:** non_disaster (p=0.03)
- **Observation:** Strong reflections (like broken glass or water reflections) may confuse the model. Needs more diverse training data.

## 5. Borderline Case – Wet Road
- **Image:** wet_road.png
- **Predicted:** non_disaster (p=0.32)
- **Observation:** Higher uncertainty. Model may confuse rain-soaked infrastructure with actual flooding.

---

### Key Takeaways
- **Text Limitations:** Struggles with figurative/sarcastic disaster terms in non-disaster contexts.
- **Image Limitations:** Struggles with reflective or water-related non-disaster scenes.
- **Future Work:** Expand adversarial training, collect edge-case datasets, and apply explainability methods (Grad-CAM, SHAP) to visualize model reasoning.

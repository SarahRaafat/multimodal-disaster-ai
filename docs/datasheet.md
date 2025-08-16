# Dataset Datasheet – Multimodal Disaster Response AI

## Sources
- **Text**: Kaggle "Disaster Tweets" dataset (https://www.kaggle.com/competitions/nlp-getting-started)
- **Images**: Flooded vs. Non-flooded drone imagery (AutoDrone Flood Dataset, Kaggle link: <insert link>)
- Both are public, research-friendly datasets with permissive licenses.

## Motivation & Intended Use
- **Goal**: Demonstrate multimodal AI for disaster detection (portfolio / internship project).
- **Intended Use**: Research demo and educational purposes only.
- **Not Intended For**: Real-world emergency response, operational deployment, or decision-making that affects lives.

## Composition
- Text: ~5k tweets, balanced between "disaster" and "non-disaster".
- Images: ~1.5k drone photos, balanced between "flooded" and "non-flooded".
- English-only text, limited geographic representation.

## Collection Process
- Text: Pre-collected from Twitter by dataset creators, already anonymized.
- Images: Curated drone photography, no personal identifiers.

## Preprocessing
- Text: Cleaned (URLs, mentions, hashtags removed), balanced, stratified train/val/test.
- Images: Resized (224×224), deduplicated, balanced, stratified train/val/test.

## Distribution of Splits
- Text: Train 4,000 / Val 500 / Test 500.
- Images: Train 1,192 / Val 149 / Test 149 (592/74/74 flooded vs. 600/75/75 non-flooded).

## Limitations
- Small dataset size; models are prototypes.
- Language bias (English-only).
- Label noise possible (tweets can be sarcastic, images mislabeled).
- Non-exhaustive disaster types (only floods).

## Ethical Considerations
- Do not use this model for real disaster response.
- Biases in data may lead to incorrect conclusions.
- Privacy: No PII present; all text anonymized by source.

## License & Access
- Kaggle datasets under permissive licenses (non-commercial research use).
- Users must respect original dataset terms.

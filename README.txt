# 🦷 Oral Diseases Image Classification

Deep learning project that classifies oral disease images into 6 categories, built during the INSTANT AI & Data Analysis training. It compares a custom CNN against pretrained models, tunes hyperparameters, and deploys the best model as a web app.

> ⚠️ Educational project only. Not a medical diagnosis tool.

## Dataset
[Oral Diseases (Kaggle)](https://www.kaggle.com/datasets/salmansajid05/oral-diseases): 6 classes.

| Class | Original images used |
|---|---|
| Gingivitis | 2349 |
| Calculus | 1296 |
| Hypodontia | 1251 |
| Mouth Ulcer | 265 |
| Caries | 219 |
| Tooth Discoloration | 183 |

**Data decisions**
- Used only the **original** images (skipped the pre-augmented folders and the YOLO-annotated folder) to avoid near-duplicate leakage between train and test.
- Stratified split: 70% train / 15% validation / 15% test.
- Class imbalance handled with class weights.
- Augmentation (flip, rotation, zoom, brightness, contrast) applied only on the training data, inside the model.

## Models
| Model | Test Accuracy | Macro F1 |
|---|---|---|
| Custom CNN (VGG-style, from scratch) | 0.60 | 0.53 |
| EfficientNetB0 (frozen) | 0.753 | 0.720 |
| MobileNetV2 (frozen) | 0.757 | 0.733 |
| **MobileNetV2 + fine-tuning** | **0.790** | **0.770** |

**Custom CNN:** 3 blocks of 2×Conv(3×3) + MaxPooling (32 → 64 → 128 filters), Dense(128), Dropout(0.5), softmax.

**Hyperparameter tuning (MobileNetV2, frozen base):**
| Learning rate | Dropout | Test Accuracy | Macro F1 |
|---|---|---|---|
| 1e-3 | 0.3 | 0.757 | 0.733 |
| 1e-3 | 0.5 | 0.749 | 0.721 |
| 1e-4 | 0.3 | 0.675 | 0.620 |
| 1e-4 | 0.5 | 0.667 | 0.611 |

Learning rate had the biggest effect; dropout had a minor one. Final step: unfroze the last 30 layers with learning rate 1e-5.

## Results & analysis
- Transfer learning clearly beat training from scratch on this small dataset (+24 points of macro F1 vs. the custom CNN).
- The biggest gain was on the small classes (Caries, Tooth Discoloration).
- Calculus is the hardest class, likely confused with Gingivitis (visually similar).

## Limitations
- Small test sets for minority classes (27–39 images), so their metrics are unstable.
- The pre-augmented folders were excluded; I assumed the remaining folders (Gingivitis, Calculus, Hypodontia) contain original images without running a duplicate check.
- Not clinically validated.

## Run the app
```bash
pip install -r requirements.txt
python app.py
```
The notebook (`notebook.ipynb`) contains the full pipeline: data preparation, training, evaluation, and the Gradio app.

## Tech stack
Python · TensorFlow/Keras · scikit-learn · Gradio · Google Colab
# Parasite Egg Classification

A computer vision pipeline for classifying microscopic parasite egg images into 11 species using YOLOv8 object detection. Built for the SuperAI Engineer Season 6 competition (Chula-ParasiteEgg-11 dataset).

## Competition Details

| Item | Value |
|---|---|
| Metric | Macro F1-score (equal weight per class) |
| Submission limit | 4 per day |
| Challenge | Train ≠ Test distribution (domain shift) |
| Open-set | Test contains unknown classes + background → label `-1` |
| Allowed | Public external data + test data for pseudo-labeling |

## Dataset

**Primary:** Chula-ParasiteEgg-11
- 11,000 images in COCO JSON format
- 11 parasite egg species
- Split: 8,800 train / 2,200 val (stratified 80/20)
- Test: 2,002 images

**External data (fine-tuning):**
- Mendeley Bangladesh dataset — ~300 negative/background images
- AI4NTD P1.5 (Kaggle: `peterkward/ai4ntd-p1-5`) — Kato-Katz smear images for Ascaris, Hookworm, Trichuris

## Classes

| ID | Species |
|---|---|
| 0 | Ascaris lumbricoides |
| 1 | Capillaria philippinensis |
| 2 | Enterobius vermicularis |
| 3 | Fasciolopsis buski |
| 4 | Hookworm egg (Necator/Ancylostoma) |
| 5 | Hymenolepis diminuta |
| 6 | Hymenolepis nana |
| 7 | Opisthorchis viverrine |
| 8 | Paragonimus spp |
| 9 | Taenia spp. egg |
| 10 | Trichuris trichiura |
| -1 | Unknown / No egg (open-set) |

## Model

**YOLOv8m** — 25.86M parameters, 79.1 GFLOPs, 640×640 input

Key training settings:
- Heavy augmentation: HSV shift, rotation 45°, scale 0.5, mosaic, mixup
- SGD optimizer, cosine LR schedule
- Per-class confidence threshold tuning (greedy Macro F1 optimization)

## Pipeline

### Phase 1 — Baseline Training
1. Convert COCO JSON → YOLO format
2. Stratified train/val split (ensures balanced class distribution in val)
3. Train YOLOv8m for 10 epochs on Chula-ParasiteEgg-11
4. Run val inference at conf=0.01, then greedy per-class threshold tuning
5. Generate 4 submission variants: optimal, conservative, aggressive, uniform

**Val Macro F1: 0.9881**

### Phase 2 — Fine-tune with Mendeley Negatives
- Adds ~300 background images (empty YOLO labels) to teach the model to output `-1`
- Fine-tune from Phase 1 checkpoint: 8 epochs, AdamW, lr=0.0003
- Produces `best_finetuned.pt`

### Phase 3 — Fine-tune with AI4NTD (Kato-Katz domain)
- Maps AI4NTD classes to Chula IDs (Ascaris→0, Hookworm→4, Trichuris→10, Schistosoma→dropped)
- Adds ~500+ Kato-Katz smear images to combat domain shift for 3 key classes
- Fine-tune from Phase 2 checkpoint: 10 epochs, AdamW, lr=0.0003
- Produces `best_ai4ntd.pt`

## Submission Strategy

Each phase generates 4 CSV variants per day budget:

| Variant | Threshold | Purpose |
|---|---|---|
| `*_v1_optimal.csv` | Per-class tuned | Main submission |
| `*_v2_conservative.csv` | +0.05 all classes | Reduce false positives |
| `*_v3_aggressive.csv` | -0.05 all classes | Increase recall |
| `*_v4_uniform.csv` | 0.25 all classes | Baseline reference |

Submission priority: `v1 (optimal)` → `v3 (aggressive)` → `v2 (conservative)` → `v4 (uniform)`

## Selecting the Best Model

After submitting all three phase variants, compare public LB scores:

| Public LB pattern | Interpretation | Next action |
|---|---|---|
| Baseline wins | External data confuses the model | Use baseline for pseudo-labeling |
| Mendeley wins | Test set has many `-1` images | Add more negative data |
| AI4NTD wins | Domain shift is the main issue | Add more external data for remaining classes |
| All similar | All data helps marginally | Ensemble the 3 models via WBF |

## Key Technical Details

**Per-class threshold tuning** — Critical for Macro F1: low-frequency classes need lower detection thresholds to avoid recall collapse. Greedy optimization runs 3 passes over all 11 classes.

**Open-set detection** — No-detection images (all class confidences below threshold) are labeled `-1`. Mendeley negatives train the model to predict nothing rather than forcing a wrong class.

**Raw prediction caching** — Val and test predictions at conf=0.01 are cached as pickle files, allowing threshold re-tuning without re-running inference.

## Requirements

```
ultralytics
scikit-learn
opencv-python
pandas
numpy
matplotlib
seaborn
tqdm
kaggle  # for Phase 3
```

## Running

The full pipeline runs in Google Colab with Google Drive. Set `DRIVE_BASE` to your Drive path containing:
- `Chula-ParasiteEgg-11/` — training data
- `test_set/test/` — competition test images
- `super-ai-engineer-season-6-parasite-eggs.zip` — competition archive

Then execute notebook cells sequentially through the 3 phases.

## File Structure (Drive)

```
ParasiteEgg_Hack/
├── Chula-ParasiteEgg-11/       Training images + labels.json
├── test_set/test/              Competition test images
├── mendeley_dataset.zip        Mendeley negatives (manual download)
├── best_finetuned.pt           Phase 2 model checkpoint
├── best_ai4ntd.pt              Phase 3 model checkpoint
├── raw_predictions_ft.pkl      Cached Phase 2 test predictions
├── raw_predictions_ai4ntd.pkl  Cached Phase 3 test predictions
└── sub_*.csv                   Submission files
```

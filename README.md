# Product Recognition

Computer-vision systems for recognizing grocery-store products, developed for the
**Image Processing and Computer Vision** course (University of Bologna).

The repository contains two self-contained assignment modules that approach product
recognition from two complementary angles:

| Module | Notebook | Task | Approach |
|---|---|---|---|
| 1 | [product_detection_sift.ipynb](product_detection_sift.ipynb) | **Detect** products on a store shelf | Classical pipeline: denoising + SIFT + homography |
| 2 | [product_classification_cnn.ipynb](product_classification_cnn.ipynb) | **Classify** a product image into one of 43 classes | Deep learning: a custom CNN, then a fine-tuned ResNet-18 |

---

## Module 1 — Shelf Product Detection (SIFT)

**Goal.** Given one reference image per product and a photo of a store shelf, locate every
instance of each product and report, for each one, its **count**, **bounding-box centre**, and
**width / height** in pixels.

- **Track A — single instance:** one occurrence per product (references `ref1–14`, scenes `scene1–5`).
- **Track B — multiple instances:** several occurrences of the same product (references `ref15–27`, scenes `scene6–12`).

### Pipeline

1. **Denoising.** Scene images are corrupted by both salt-and-pepper and Gaussian noise.
   A **Median filter** removes the impulse noise first (so it isn't smeared by smoothing),
   followed by a **Bilateral filter** that suppresses Gaussian noise while preserving edges.
   Filter type and strength were selected with a **grid search** that maximizes good SIFT
   matches (Non-Local Means was also tested, but gave fewer good matches).
2. **Feature extraction.** **SIFT** keypoints and descriptors — invariant to scale and rotation.
3. **Matching.** Brute-force matcher with **Lowe's ratio test** (0.7) to keep distinctive matches.
4. **Localization.** **Homography** (RANSAC) + perspective transform maps the reference's
   corners into the scene, yielding the bounding box.
5. **Validation.**
   - A **dynamic threshold** requires a minimum percentage of the reference's keypoints to match.
   - **Template matching** (`TM_CCORR_NORMED`) confirms borderline detections.
   - *(Track B)* an **aspect-ratio check** and an **HSV colour check** reject false matches, and
     each detected instance is masked out (black box) so it is not detected twice.

### Notes & limitations

- SIFT operates on the grayscale image, so visually similar products of different colour can be
  confused (e.g. `ref27` vs `ref26`); the HSV check mitigates this but is hampered by image noise.
- This notebook is set up for **Google Colab**: it mounts Google Drive and unzips a `dataset.zip`
  containing `dataset/models/` (`ref1–27.png`) and `dataset/scenes/` (`scene1–12.png`). To run it
  elsewhere, replace the first data cell with a local path to that `dataset/` folder.

---

## Module 2 — Product Classification (CNN)

**Goal.** Classify natural smartphone photos from the
[GroceryStoreDataset](https://github.com/marcusklasson/GroceryStoreDataset) into **43 classes**
(fruit, vegetables, dairy/juice cartons).

### Data & preprocessing

- **Splits.** The dataset ships with `train` / `val` / `test` splits. A class-distribution plot shows
  they are **imbalanced** and that the small validation split under-represents a few rare classes —
  which is why **per-class accuracy** (not just the overall number) is reported for the best models.
- **Augmentation (train).** Random crop to **224×224** + random horizontal flip. 224 is ResNet-18's
  native ImageNet resolution, so the same inputs serve both Part 1 and Part 2; the source photos
  (~348²) are larger, so cropping preserves native detail and adds translation jitter.
- **Evaluation (val/test).** Deterministic **centre crop** to 224×224.
- **Normalization.** Inputs are normalized with **ImageNet mean/std**, the distribution the
  pretrained ResNet-18 backbone expects (the from-scratch models share the same dataloaders).

### Part 1 — A custom CNN, built step by step

Rather than dropping in a ready-made network, the architecture is grown one component at a time,
**keeping each change only if it improves validation accuracy** and otherwise discarding it with an
explanation. The baseline shared by every model is: **VGG-style stages** (`Conv 3×3 → BatchNorm →
ReLU`, channels doubling 64→512 as resolution halves), a **global-average-pooling head**
(Network-in-Network / Inception style, to avoid a fully-connected parameter explosion), the
**Adam + OneCycleLR** recipe, and **label smoothing (0.1)**.

| Model | Change | Params | Val acc | Test acc | Outcome |
|---|---|---|---|---|---|
| 1 | Single conv stage (baseline) | 41 k | 0.568 | 0.613 | start |
| 2 | + second stage | 260 k | 0.652 | 0.739 | depth helps ✓ |
| 3 | + third stage | 1.75 M | 0.689 | 0.770 | depth helps ✓ |
| 4 | Deeper (11 layers) | 7.6 M | 0.632 | 0.732 | regression ✗ |
| 5 | + dropout (test: overfitting?) | 7.6 M | 0.581 | 0.697 | hypothesis rejected ✗ |
| 6 | + residual connections (test: gradient flow?) | 11 M | **0.737** | **0.798** | **best ✓** |

The regression at Model 4 is diagnosed explicitly: dropout (Model 5) tests — and rejects — the
*overfitting* hypothesis, pointing instead to an *optimization* problem that **residual skip
connections** (Model 6, via 1×1 projection shortcuts) then fix.

**Ablation** on the best model confirms the baseline choices: removing batch normalization or the
OneCycleLR scheduler each drops validation accuracy materially.

> ℹ️ The figures above are from a **reference run** and are illustrative. The notebook now **builds
> its summary and ablation tables automatically** from the recorded results (`MODEL_RESULTS`), so the
> authoritative numbers are whatever the latest run produces — and they shift slightly after the
> recent preprocessing changes (ImageNet normalization, consistent `padding=1`). The qualitative
> arc (depth helps → plain depth stalls → residual connections recover) is the point.

### Part 2 — Transfer learning with ResNet-18

A **ResNet-18 pretrained on ImageNet-1K** is fine-tuned in two steps:

1. **Head only** (backbone frozen): ~**0.84** validation accuracy — the pretrained features already
   transfer strongly to grocery products.
2. **Full fine-tuning** with a **discriminative (sliced) learning rate** — lower LR for early,
   general layers; higher LR for late, task-specific layers (Fast.ai style): ~**0.89** validation /
   ~**0.90** test accuracy.

### Diagnostics & visualizations

The notebook is instrumented so each claim is visible, not just asserted:

- **Training curves** (loss & accuracy) with the **learning-rate schedule overlaid** on a twin axis,
  so the OneCycleLR warmup/decay can be read directly against the loss it drives.
- **Class-distribution** bar chart across the three splits.
- **Per-class accuracy** (sorted, class-named) and a **43×43 row-normalized confusion matrix**
  (each true-class row sums to 1, so errors aren't masked by class imbalance) for the two best
  models (Model 6 and the fine-tuned ResNet-18).
- A **model-comparison** overlay of all six architectures' validation curves, plus the
  auto-generated **summary** and **ablation-impact** tables.
- A **prediction grid** on random test images (green = correct, red = wrong).

---

## Running Module 2 locally (Apple Silicon)

The classification notebook has been adapted to run **locally on Apple Silicon (M-series) Macs**: it
selects the **MPS** GPU backend automatically (falling back to CUDA, then CPU), clones the dataset,
and saves weights to a local `weights/` directory.

```bash
# 1. Install dependencies (recent torch/torchvision include the MPS backend)
pip install torch torchvision numpy pandas seaborn matplotlib tqdm pillow ipywidgets

# 2. Open the notebook and run top to bottom
jupyter notebook product_classification_cnn.ipynb
```

The GroceryStoreDataset is cloned automatically on first run. Training on a laptop GPU is slow
compared to Colab — to verify the full flow quickly, temporarily lower `EPOCHS` in the training
cells, then raise it again for a real run.

**Run top to bottom and keep outputs.** The summary/ablation **tables** and all **figures** are
rendered at runtime (the tables from `MODEL_RESULTS`), so they only appear once the cells they
depend on have run, and are not stored if you clear outputs. The per-model "Result" write-ups are
plain markdown and always visible. Export/commit the notebook **with outputs** so the figures are
visible without re-running.

---

## Repository structure

```
Product-Recognition/
├── product_detection_sift.ipynb        # Module 1 — shelf detection (SIFT)
├── product_classification_cnn.ipynb    # Module 2 — classification (CNN + ResNet-18)
├── README.md
└── .gitignore
```

Datasets (`GroceryStoreDataset/`, `dataset/`) and model weights (`weights/`, `*.pth`) are generated
at runtime and are intentionally git-ignored.
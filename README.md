# Error Bar Detection in Scientific Charts

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white" alt="HuggingFace">
  <img src="https://img.shields.io/badge/Platform-Kaggle_T4_GPU-20BEFF?logo=kaggle&logoColor=white" alt="Kaggle">
</p>

<p align="center">
  <strong>Automatic detection and measurement of error bars in scientific plot images</strong>
</p>

---

## Project Background

This project was completed as a two-phase assignment:

| Phase | Task | Description |
|:-----:|------|-------------|
| **1** | Synthetic Data Generation | Generated 3,000 synthetic scientific plots with ground truth error bar annotations |
| **2** | Error Bar Detection | Developed detection pipelines to automatically measure error bars from plot images |

**Dataset:** [Google Drive - Synthetic Plots Dataset](https://drive.google.com/drive/folders/1Xm35AgzmbG1gSHRStqrMRqYmUwfLfbSo)

---

## Task Definition

| Component | Description |
|-----------|-------------|
| **Input** | Scientific plot image + Data point coordinates (x, y) in pixels |
| **Output** | `topBarPixelDistance` and `bottomBarPixelDistance` in pixels |

---

## Project Structure

```
Error-Bar-Detection-Project/
|
|-- README.md
|-- .gitignore
|-- figure 1.png                              # Pipeline 1 architecture
|-- figure 2.png                              # Pipeline 2 architecture
|
|-- Basic OpenCV + ROI/
|   |-- OpenCV + ROI.ipynb
|   |-- Results/
|       |-- summary_metrics.csv
|       |-- per_image_metrics.csv
|       |-- predictions/
|
|-- OpenCV + ROI + ML Refinement/
|   |-- cv-ml-hybrid-errorbar-detection-new.ipynb
|   |-- Results/
|       |-- final_model_comparison_600samples.csv
|       |-- ablation_study.csv
|       |-- feature_descriptions.csv
|       |-- predictions/
|
|-- Zero shot QWEN2.5-VL-7B base & finetuned/
|   |-- Zero shot QWEN2.5-VL-7B inference.ipynb
|   |-- qwen2-5-vl-error-bar-detection-fine-tuning.ipynb
|   |-- Chartqwen inference.ipynb
|   |-- Results of base QWEN2.5-VL-7B-INSTRUCT/
|   |   |-- qwen_vqa_summary_metrics.csv
|   |   |-- qwen_vqa_per_image_metrics.csv
|   |   |-- predictions/
|   |-- Results of Chartqwen/
|       |-- chartqwen_summary_metrics.csv
|       |-- chartqwen_per_image_metrics.csv
|       |-- predictions/
```

---

## Methodology

This project implements two distinct pipelines for error bar detection, each with different trade-offs between speed, accuracy, and computational requirements.

---

### Pipeline 1: CV + ROI with ML Refinement

<p align="center">
  <img src="figure 1.png" alt="Pipeline 1 Architecture" width="500">
</p>
<p align="center"><em>Figure 1: Computer Vision + Machine Learning Hybrid Pipeline</em></p>

#### Architecture Overview

```
Input Image + Coordinates --> ROI Extraction --> Edge Detection --> Vertical Line Analysis
                                   |                   |                    |
                                   v                   v                    v
                            Feature Extraction   Canny Edges         Line Projection
                                   |                   |                    |
                                   +-------------------+--------------------+
                                                       |
                                                       v
                                              CV Coarse Estimate
                                                       |
                                                       v
                                              ML Refinement (MLP)
                                                       |
                                                       v
                                              Final Prediction
```

#### Stage 1: Computer Vision Baseline

| Step | Operation | Parameters |
|------|-----------|------------|
| 1 | ROI Extraction | 30px width x 150px height around data point |
| 2 | Grayscale Conversion | BGR to single channel |
| 3 | Edge Detection | Canny with thresholds (50, 150) |
| 4 | Vertical Line Detection | Min length 5px, tolerance 3px |
| 5 | Distance Calculation | Geometric measurement to endpoints |

**CV Parameters:**
```python
ROI_WIDTH = 30          # Horizontal search region
ROI_HEIGHT = 150        # Vertical search region
EDGE_THRESHOLD1 = 50    # Canny lower threshold
EDGE_THRESHOLD2 = 150   # Canny upper threshold
MIN_LINE_LENGTH = 5     # Minimum error bar segment
VERTICAL_TOLERANCE = 3  # Max x-axis deviation
```

#### Stage 2: ML Refinement

**Feature Categories (14 total):**

| Category | Features | Description |
|----------|----------|-------------|
| ROI Intensity | `roi_mean_intensity`, `roi_std_intensity`, `roi_min_intensity`, `roi_max_intensity` | Pixel brightness statistics |
| ROI Gradients | `roi_grad_y_mean`, `roi_grad_y_std` | Vertical gradient strength |
| ROI Edges | `roi_edge_density` | Percentage of edge pixels |
| ROI Center | `roi_center_col_mean`, `roi_center_col_std`, `roi_center_col_min`, `roi_center_col_max` | Center column statistics |
| CV Coarse | `cv_coarse_top_dist`, `cv_coarse_bottom_dist`, `cv_confidence` | CV detection outputs |

**ML Models Evaluated:**

| Model | Architecture | Hyperparameters |
|-------|--------------|-----------------|
| Linear Regression | Single layer | Default |
| Ridge Regression | L2 regularized | alpha=1.0 |
| MLP | 3 hidden layers | (64, 32, 16), ReLU |
| Tuned MLP | Grid-searched MLP | Optimized hidden sizes |
| Random Forest | Ensemble trees | 100 estimators, max_depth=10 |
| Gradient Boosting | Sequential trees | 100 estimators, lr=0.1 |

#### Why Hybrid Approach?

| Pure CV Limitations | Pure ML Limitations | Hybrid Advantages |
|---------------------|---------------------|-------------------|
| Fixed thresholds fail on varied styles | Requires massive labeled data | CV provides geometric structure |
| Noise sensitive edge detection | Black box predictions | ML refines CV errors |
| Cannot adapt to dataset patterns | Overfits to artifacts | Data efficient training |
| No learning capability | No geometric understanding | Explainable pipeline |

---

### Pipeline 2: Vision-Language Model Approach

<p align="center">
  <img src="figure 2.png" alt="Pipeline 2 Architecture" width="650">
</p>
<p align="center"><em>Figure 2: Vision-Language Model Pipeline</em></p>

#### Architecture Overview

```
Input Image + Coordinates --> Image Encoder --> Vision Tokens
                                                     |
                                                     v
User Prompt (with coordinates) --> Tokenizer --> Text Tokens
                                                     |
                                                     +-----> Multimodal Fusion
                                                                  |
                                                                  v
                                                     Language Model Decoder
                                                                  |
                                                                  v
                                                     JSON Output Parser
                                                                  |
                                                                  v
                                                     Pixel Measurements
```

#### Model Configuration

**Base Model: Qwen2.5-VL-7B-Instruct**

| Specification | Value |
|--------------|-------|
| Architecture | Vision-Language Transformer |
| Parameters | 7 Billion |
| Vision Encoder | ViT-based |
| Context Length | 32K tokens |
| Image Resolution | Dynamic (up to 768px) |

**Inference Configuration:**

```python
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.1
IMAGE_MAX_SIZE = 768
PRECISION = "float16"  # FP16 for T4 GPU
```

#### Fine-tuning: Chartqwen

The base model is fine-tuned using LoRA (Low-Rank Adaptation) for task-specific performance.

**LoRA Configuration:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| Rank (r) | 32 | Low-rank decomposition dimension |
| Alpha | 64 | Scaling factor |
| Dropout | 0.05 | Regularization |
| Target Modules | q_proj, k_proj, v_proj, o_proj | Attention layers |

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Training Samples | 600 |
| Epochs | 1 |
| Batch Size | 1 |
| Gradient Accumulation | 16 |
| Effective Batch Size | 16 |
| Learning Rate | 2e-4 |
| Warmup Ratio | 0.03 |
| Max Gradient Norm | 1.0 |

**Published Model:** [Sayeem26s/Chartqwen](https://huggingface.co/Sayeem26s/Chartqwen)

#### Prompt Engineering

**System Prompt:**
```
You are a precise error bar detection system for scientific plots.
Given an image of a scientific plot and data point coordinates, detect the error bars.
For each point, output the pixel distance from the data point to the top and bottom of the error bar.
If no error bar exists for a point, output 0 for both distances.
```

**User Prompt Template:**
```
Analyze this scientific plot image and detect error bars for the following data points:

[{"x": 96.6, "y": 70.9}, {"x": 120.3, "y": 85.2}, ...]

For each point, measure:
- topBarPixelDistance: pixel distance from data point to top of error bar (0 if none)
- bottomBarPixelDistance: pixel distance from data point to bottom of error bar (0 if none)

Output as JSON array:
[
  {"x": <x>, "y": <y>, "topBarPixelDistance": <top>, "bottomBarPixelDistance": <bottom>}
]
```

---

## Results Summary

### All Methods Comparison

| Rank | Method | Test Set | Mean Error (px) | RMSE (px) | Acc@5px | Acc@10px | Acc@20px |
|:----:|--------|:--------:|:---------------:|:---------:|:-------:|:--------:|:--------:|
| 1 | **CV Only (Baseline)** | 600 img / 10,229 pts | **21.07** | 37.74 | 0.5% | 10.5% | **72.7%** |
| 2 | MLP Refinement | 600 img / 10,229 pts | 23.93 | 36.36 | 0.0% | 2.7% | 52.3% |
| 3 | Tuned MLP | 600 img / 10,229 pts | 24.04 | 36.26 | 0.1% | 2.3% | 51.0% |
| 4 | Linear Regression | 600 img / 10,229 pts | 24.79 | 36.70 | 0.1% | 1.0% | 44.2% |
| 5 | Ridge Regression | 600 img / 10,229 pts | 24.79 | 36.70 | 0.1% | 1.0% | 44.2% |
| 6 | Gradient Boosting | 600 img / 10,229 pts | 27.67 | 56.71 | 0.1% | 6.3% | 65.3% |
| 7 | Random Forest | 600 img / 10,229 pts | 30.26 | 70.28 | 0.4% | 8.9% | 67.0% |
| 8 | Basic CV + ROI | 600 img / 10,229 pts | 34.38 | 52.50 | 16.8% | 29.4% | 44.3% |
| 9 | Qwen2.5-VL (Zero-shot) | 600 img / 10,229 pts | 40.22 | 67.77 | 17.5% | 27.2% | 44.1% |
| 10 | **Chartqwen (Fine-tuned)** | 100 img / 784 pts | 41.75 | 71.06 | 9.1% | 22.2% | 40.9% |

---

## Detailed Results

### Pipeline 1: CV + ML Hybrid Results

#### Baseline vs Refinement Comparison

| Approach | Mean Error (px) | Baseline (px) | Improvement (%) | Error Reduction (px) |
|----------|:---------------:|:-------------:|:---------------:|:--------------------:|
| CV Only (Baseline) | 21.07 | - | - | - |
| MLP | 23.93 | 21.07 | -13.5 | -2.85 |
| Tuned MLP | 24.04 | 21.07 | -14.1 | -2.96 |
| Ridge Regression | 24.79 | 21.07 | -17.7 | -3.72 |
| Linear Regression | 24.79 | 21.07 | -17.7 | -3.72 |
| Gradient Boosting | 27.67 | 21.07 | -31.3 | -6.59 |
| Random Forest | 30.26 | 21.07 | -43.6 | -9.18 |

#### Ablation Study

| Method | MAE Top (px) | MAE Bottom (px) | MAE Avg (px) | RMSE Avg (px) |
|--------|:------------:|:---------------:|:------------:|:-------------:|
| CV Only | 29.34 | 27.12 | 28.23 | 223.50 |
| ML Only (no CV) | 32.66 | 32.06 | 32.36 | 183.59 |
| Hybrid (CV + MLP) | 31.49 | 28.77 | 30.13 | 221.87 |

### Pipeline 2: VLM Results

#### Qwen2.5-VL Zero-Shot Performance (600 Images)

| Metric | Value |
|--------|:-----:|
| Total Images | 600 |
| Total Points | 10,229 |
| Mean Top Error | 42.40 px |
| Mean Bottom Error | 38.04 px |
| Mean Overall Error | 40.22 px |
| RMSE | 67.77 px |
| Accuracy @ 5px | 17.50% |
| Accuracy @ 10px | 27.18% |
| Accuracy @ 20px | 44.09% |

#### Chartqwen Fine-tuned Performance (100 Images)

| Metric | Value |
|--------|:-----:|
| Total Images | 100 |
| Total Points | 784 |
| Mean Top Error | 41.76 px |
| Mean Bottom Error | 41.75 px |
| Mean Overall Error | 41.75 px |
| RMSE | 71.06 px |
| Accuracy @ 5px | 9.06% |
| Accuracy @ 10px | 22.19% |
| Accuracy @ 20px | 40.94% |

### Basic CV + ROI Results (600 Images)

| Metric | Value |
|--------|:-----:|
| Total Images | 600 |
| Total Points | 10,229 |
| Mean Top Error | 34.81 px |
| Mean Bottom Error | 33.95 px |
| Mean Overall Error | 34.38 px |
| Median Overall Error | 24.99 px |
| RMSE | 52.50 px |
| Accuracy @ 5px | 16.84% |
| Accuracy @ 10px | 29.35% |
| Accuracy @ 20px | 44.27% |

---

## Key Findings

| Finding | Details |
|---------|---------|
| Best Overall | CV-only baseline achieves lowest mean error (21.07px) and highest Acc@20px (72.7%) |
| ML Refinement | Does not improve over CV baseline; increases error by 13-44% |
| VLM Zero-shot | Comparable to basic CV but significantly slower (40.22px mean error) |
| Fine-tuning Impact | Chartqwen shows similar performance to base model on smaller test set |
| Speed vs Accuracy | CV pipeline is ~100x faster than VLM approach |

---

## Running the Notebooks

All notebooks run on **Kaggle Free Tier T4 GPU** (30 hrs/week quota).

| Notebook | GPU | Runtime |
|----------|:---:|:-------:|
| OpenCV + ROI.ipynb | CPU | ~10 min |
| cv-ml-hybrid-errorbar-detection-new.ipynb | CPU | ~15 min |
| Zero shot QWEN2.5-VL-7B inference.ipynb | T4 | ~45 min |
| qwen2-5-vl-error-bar-detection-fine-tuning.ipynb | T4 | ~1.5 hr |
| Chartqwen inference.ipynb | T4 | ~3 min |

---

## Model Artifacts

### Published Model

| Model | Hub | Link |
|-------|-----|------|
| Chartqwen | HuggingFace | [Sayeem26s/Chartqwen](https://huggingface.co/Sayeem26s/Chartqwen) |

### Loading Chartqwen

```python
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "Sayeem26s/Chartqwen")
model = model.merge_and_unload()  # Merge for faster inference

# Load processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
```

---

## References

### Models

- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) - Base vision-language model
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685) - Parameter-efficient fine-tuning

### Libraries

- [OpenCV](https://opencv.org/) - Computer vision operations
- [Transformers](https://huggingface.co/docs/transformers) - Model loading and inference
- [PEFT](https://huggingface.co/docs/peft) - Parameter-efficient fine-tuning
- [scikit-learn](https://scikit-learn.org/) - Machine learning models

---

<p align="center">
  <strong>Developed for automated scientific chart analysis</strong>
</p>

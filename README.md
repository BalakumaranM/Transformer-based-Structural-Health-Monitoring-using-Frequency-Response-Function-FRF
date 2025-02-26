---

# Beam Damage Assessment Using FRF Data

This repository contains all the code and resources for a beam damage detection project that uses Frequency Response Function (FRF) data, FRAC-based features, and a hybrid CNN + Transformer model for classification. The dataset is sourced from [Zenodo (Record \#8081690)](https://zenodo.org/records/8081690).

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Dataset](#dataset)  
3. [Repository Structure](#repository-structure)  
4. [Installation & Setup](#installation--setup)  
5. [Usage](#usage)  
   - [Training](#training)  
   - [Inference](#inference)  
   - [Confusion Matrix Plots](#confusion-matrix-plots)  
6. [Blog Posts](#blog-posts)  
7. [License](#license)  
8. [References](#references)

---

## Project Overview

The goal of this project is to **detect damage in a cantilever beam** reinforced with masses by analyzing vibration signals in the frequency domain. We use:

- **FRF Data** (Frequency, Magnitude in dB, Phase)  
- **FRAC** (Frequency Response Assurance Criterion) to compute per-frequency correlation with a healthy reference  
- **Hybrid CNN+Transformer** architecture to learn from the sequence data and classify the beam condition into four classes:  
  1. Healthy  
  2. Damage 2.96%  
  3. Damage 5.92%  
  4. Damage 8.87%

The repository includes **training scripts**, **inference scripts**, and code for **plotting confusion matrices** (in both raw counts and percentage-of-total form).

---

## Dataset

We use the dataset from **[Zenodo (Record \#8081690)](https://zenodo.org/records/8081690)**:  
> Amanda A.S.R. de Sousa, Marcela R. Machado, *Experimental vibration dataset collected of a beam reinforced with masses under different health conditions*, Data in Brief, 2024.  

**Key Points**:
- Four folders, each representing a structural condition:  
  1. Beam_healthy  
  2. Beam_damage_2.96  
  3. Beam_damage_5.92  
  4. Beam_damage_8.87  
- Each folder has 70 text files with columns: Frequency, Magnitude, Phase.  
- FRAC is used to compare each damaged sample with its corresponding healthy sample.

You should **download** or **clone** this dataset and organize it in a structure like:

```
Data/
  ├─ Beam_healthy/
  │   ├─ P-1.txt
  │   ├─ P-2.txt
  │   └─ ...
  ├─ Beam_damage_2.96/
  │   ├─ P-1.txt
  │   ├─ P-2.txt
  │   └─ ...
  ├─ Beam_damage_5.92/
  ├─ Beam_damage_8.87/
```

---

## Repository Structure

Below is a typical structure you might use:

```
.
├─ Data/                          # Place the Zenodo dataset here
├─ best_models/                   # Where your best model weights are saved
├─ train.py             # Main training pipeline
├─ inference.py                   # Inference and evaluation script
├─ README.md                      # This README file
└─ ...
```

---

## Installation & Setup

1. **Clone or Download** this repository.
2. **Install Python 3.8+** (e.g., using [pyenv](https://github.com/pyenv/pyenv) or your system’s package manager).
3. **Create a virtual environment** and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Linux/macOS
   # Or "venv\Scripts\activate" on Windows

   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Organize the dataset** from [Zenodo (Record \#8081690)](https://zenodo.org/records/8081690) into `Data/`.

---

## Usage

### Training

1. **Set hyperparameters** in `training_script.py` (e.g., `freq_max`, `downsample_factor`, `smoothing`, etc.).  
2. **Run**:
   ```bash
   python train.py
   ```
3. **Outputs**:
   - A model checkpoint, e.g., `best_hybrid_model.pth`, stored in `./best_models/`.

### Inference

1. **Adjust** any inference parameters in `inference.py` (e.g., `downsample_factor`, `smoothing`, etc.).  
2. **Run**:
   ```bash
   python inference.py
   ```
3. **Outputs**:
   - Test accuracy, classification report, confusion matrices (raw counts and percentages), saved in `training_graph/`.

### Confusion Matrix Plots

- Two confusion matrices are generated:
  1. **Raw Counts**: `confusion_matrix_counts_<timestamp>.png`
  2. **Percentage of Total**: `confusion_matrix_totalperc_<timestamp>.png`

Both are saved under `training_graph/`.

---

## Blog Posts

To learn more about the background, methodology, and advanced analysis, see the following blog posts:

- [**Blog Post 1: Project Introduction**](https://yourblog.example.com/beam-damage-intro)  
- [**Blog Post 2: Data Preprocessing & FRAC**](https://yourblog.example.com/beam-damage-preprocessing)  
- [**Blog Post 3: CNN+Transformer Architecture**](https://yourblog.example.com/beam-damage-cnn-transformer)  
- [**Blog Post 4: Final Results & Discussion**](https://yourblog.example.com/beam-damage-final-results)

Each post dives deeper into the project’s motivation, the dataset, and step-by-step code explanations.

---

## License

[MIT License](LICENSE) or the license of your choosing.

---

## References

1. **Zenodo Dataset**: [Amanda A.S.R. de Sousa, Marcela R. Machado, *Experimental vibration dataset of a beam reinforced with masses*, 2024.](https://zenodo.org/records/8081690)  
2. **Relevant Paper**:  
   - Amanda A.S.R. de Sousa, Marcela R. Machado, *Multiclass supervised machine learning algorithms applied to damage and assessment using beam dynamic response*, *Journal of Vibration Engineering & Technologies* (2023).  
3. **FRAC**: Frequency Response Assurance Criterion, a standard technique for comparing FRF signals.  

---

**Happy coding and research!** If you have any questions, open an issue or reach out via the contact info on the blog.
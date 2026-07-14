# Predicting Student Performance — A Comparative Study of Dimensionality Reduction and Classification Methods

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-scientific-013243?logo=numpy&logoColor=white">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-dataframes-150458?logo=pandas&logoColor=white">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

> End-to-end machine learning project that predicts a student's final-grade band from demographic, social and school-related features, and — more importantly — **quantifies how much the choice of feature representation and model actually matters** once you evaluate honestly.

Academic project for the *Machine Learning and Artificial Intelligence* course, reworked as a self-contained, reproducible case study. Every modelling choice is stated, justified, and tested against a baseline and against sampling noise, rather than reported as a single lucky accuracy number.

---

## Highlights

- Built an **end-to-end ML pipeline** on 649 student records and 43 features after encoding, fully reproducible (`random_state=42`) and leakage-free.
- **Identified and removed severe target leakage** (`G1`/`G2`, Pearson $r \approx 0.83$–$0.92$ with the final grade) that trivially inflates scores in many versions of this dataset.
- **Evaluated 5 classifier families across 3 feature representations** (original 43-D, unsupervised PCA, supervised Fisher/MDA), applying an instructor-provided Fisher MDA class alongside the scikit-learn stack.
- Performed **hyperparameter selection with Grid Search + stratified 5-fold cross-validation**, standardizing inside each fold to avoid leakage.
- **Quantified statistical significance**: reported 95% confidence intervals and showed the leading models' differences fall within sampling noise.
- Grounded every empirical result in the underlying **theory** (why QDA overfits, why k-NN suffers the curse of dimensionality, why supervised MDA beats unsupervised PCA).

---

## At a Glance

- **Task.** Multiclass classification of students into **Low / Medium / High** final-grade bands (Portuguese-course subset of the UCI *Student Performance* dataset, 649 students, 33 raw attributes → 43 features after encoding).
- **Pipeline.** Clean encoding → standardization → three feature representations (**full space**, **PCA @ 95% variance**, **Fisher/MDA 2-D**) → five classifier families (**LDA, QDA, linear & RBF SVM, k-NN, MLP**).
- **Evaluation.** Model selection with **Grid Search + 5-fold CV**, reporting on **balanced metrics** and **stratified k-fold cross-validation with 95% confidence intervals** — not a single train/test split.
- **Headline finding.** On this genuinely noisy dataset the classes overlap heavily; the top models cluster together and their **confidence intervals overlap**, so the "winner" on a single split is within sampling noise. The interpretable linear models (LDA / linear SVM) stay competitive with the RBF SVM and the MLP, while **QDA and k-NN degrade as predicted by theory** (unstable covariance estimates and the curse of dimensionality). The project's value is in *showing why*, not in chasing a headline accuracy.

---

## Why this project is worth a look

Most student ML projects report one accuracy figure and stop. This one is built around the questions an engineer or researcher actually asks:

1. **Is the signal real?** Every result is compared to a majority-class **baseline (~39.8%)** and to the spread you get from resampling. Improvements that fall inside the noise band are reported as such.
2. **Did we leak the answer?** The intermediate grades `G1` and `G2` are almost perfect proxies for the final grade `G3`. They are **explicitly dropped** to avoid target leakage — a subtle mistake that silently inflates scores in many versions of this dataset.
3. **Does dimensionality reduction help or just look nice?** The same classifiers are run on the full 43-D space, on an **unsupervised** projection (PCA), and on a **supervised** one (Fisher/MDA), with an explicit note that MDA already "borrows" the labels — so its apparent gains must be read with care.
4. **Theory ↔ practice.** Each method is derived from first principles (PCA via both maximum-variance and minimum-reconstruction-error formulations and its link to the SVD; LDA/QDA from a Bayesian argument; the SVM primal/dual and hard vs. soft margin; the curse of dimensionality with a closed-form nearest-neighbour distance) and then the empirical results are read *against* that theory.

---

## Dataset

**UCI Student Performance** — Portuguese-language course ([UCI ML Repository #320](https://archive.ics.uci.edu/dataset/320/student+performance)).

| Property | Value |
|---|---|
| Observations | 649 students (Gabriel Pereira & Mousinho da Silveira schools, 2005–2006) |
| Raw attributes | 33 (demographic, social, school-related) |
| Features after encoding | 43 (label + one-hot encoding of categoricals) |
| Missing values | none |
| Target | `G3` final grade (0–20), discretized into 3 balanced bands at empirical quantiles (~30 / 40 / 30%) |

A binary variant (**pass** `G3 ≥ 10` vs **fail**) is also used for the one-axis Fisher analysis. The Portuguese subset was chosen over the Mathematics one because its larger sample gives more stable estimates and a more reliable train/test split.

---

## Methodology

```
raw CSV
  └─ drop G1, G2                     # prevent target leakage
  └─ label + one-hot encoding        # 43 features
  └─ StandardScaler                  # zero-mean / unit-variance
        │
        ├─ Full space (43-D)
        ├─ PCA  → 95% explained variance   (unsupervised)
        └─ Fisher / MDA → 2-D              (supervised)
                    │
   ┌────────────────┼───────────────────────────────┐
  LDA / QDA      linear & RBF SVM               k-NN     MLP
                (Grid Search + 5-fold CV)
                    │
        Comparison across all 3 representations
        → balanced metrics + stratified k-fold CV (95% CI)
```

**Techniques implemented and analyzed**

- **PCA** — maximum-variance and minimum-reconstruction-error formulations, SVD connection, scree plot / explained-variance thresholding, score & loading plots.
- **Fisher Discriminant Analysis (FDA/MDA)** — two-class and multiclass cases, projection vs. classification (LDA), geometric interpretation; built on the instructor-provided `MultipleFisherDiscriminantAnalysis` class (`FisherDA.py`), applied and analyzed throughout.
- **LDA & QDA** — derived as Bayesian classifiers under shared vs. per-class covariance; used to explain QDA's overfitting in high dimensions.
- **Support Vector Machines** — primal/dual formulation, support vectors, hard vs. soft margin, linear and RBF kernels, hyperparameter tuning via Grid Search + cross-validation, decision-boundary visualization in the LDA 2-D projection.
- **k-Nearest Neighbors** — used as a probe for the curse of dimensionality.
- **Multi-Layer Perceptron** — activation choice (ReLU), cost function, back-propagation, and hyperparameter selection via Grid Search + CV.
- **Robust evaluation** — class-balance handling, balanced accuracy / per-class metrics, stratified k-fold cross-validation with confidence intervals.

The near-flat PCA spectrum (no dominant components → 34 of 43 PCs needed for 95% variance) and the difference between the unsupervised PCA projection and the supervised Fisher/MDA one:

<p align="center">
  <img src="figures/fig_pca_scree_plot.png" alt="PCA scree plot" width="46%">
  <img src="figures/fig_mda_vs_pca_multiclasse.png" alt="PCA vs MDA projections" width="52%">
</p>

---

## Results

**Robust comparison — stratified 5-fold cross-validation** (standardized 43-D space, majority-class baseline = **39.75%**):

| Model | Mean accuracy | Std dev | 95% CI |
|---|:---:|:---:|:---:|
| **SVM** (RBF) | **56.85%** | 3.98 | [51.3, 62.4] |
| MDA / LDA | 56.69% | 4.81 | [50.0, 63.4] |
| MLP | 55.00% | 2.62 | [51.4, 58.6] |
| k-NN | 50.23% | 2.09 | [47.3, 53.1] |
| QDA | 46.38% | 4.28 | [40.4, 52.3] |

The two leading models' confidence intervals **overlap** — their difference is **not statistically significant**. On the single 75/25 split the best single result is **k-NN = 59.51%** on the 2-D supervised MDA space, but with ~163 test samples ($\pm$7–8 pts of sampling error) that ranking is within the noise.

**Effect of the feature representation** (best test accuracy per model, %):

| Model | Original (43-D) | PCA (95%, 34-D) | Fisher/MDA (2-D) |
|---|:---:|:---:|:---:|
| MDA / LDA | 55.83 | 54.60 | 55.83 |
| QDA | 40.49 | 53.37 | 57.06 |
| SVM | 54.60 | 52.15 | 57.06 |
| k-NN | 53.37 | 49.08 | **59.51** |
| MLP | 53.99 | 49.69 | 58.28 |

<p align="center">
  <img src="figures/fig_confronto_modelli.png" alt="Model comparison across feature spaces" width="70%">
</p>

**Takeaways**

- On overlapping, noisy classes, **model complexity buys little**: linear boundaries (LDA, linear SVM) perform on par with the RBF SVM and the MLP.
- **QDA collapses on the raw space** (40.49%, barely above baseline) — estimating a full covariance per class on a few hundred samples in 43 dimensions is unstable — but recovers to 57% once the dimensionality is reduced.
- **k-NN underperforms on the raw space** because Euclidean distances concentrate in high dimensions, then jumps +6.1 points on the supervised 2-D MDA projection.
- **Supervised MDA (2-D) beats unsupervised PCA (34-D)**: two label-aware axes preserve separability better than 34 variance-only ones — the empirical proof of the PCA-vs-MDA distinction.

---

## Repository structure

A single, self-contained folder (a notebook-centred academic project, not a production service):

```
.
├── eng_student_performance_ml_pipeline.ipynb   # main notebook — English (theory + code + analysis)
├── eng_student_performance_ml_pipeline.html	# export as html
├── ita_student_performance_ml_pipeline.ipynb   # original Italian version (course submission)
├── ita_student_performance_ml_pipeline.html	# export as html
├── ita_student_performance_ml_pipeline.pdf     # rendered export of the notebook (Italian)
├── FisherDA.py                                 # instructor-provided Fisher/MDA class (scikit-learn-style API)
├── student-por.csv                             # dataset (Portuguese course)
├── student.txt                                 # dataset attribute description / project notes
├── figures/                                    # plots generated by the notebook
├── LICENSE
└── README.md
```

> The notebook is the single source of truth; the English and Italian versions produce identical numeric results (`random_state=42`).

---

## Getting started

```bash
# 1. clone
git clone https://github.com/GiorgiaColucci/supervised-unsupervised-student-performance.git
cd supervised-unsupervised-student-performance

# 2. environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib scipy jupyter

# 3. run
jupyter notebook eng_student_performance_ml_pipeline.ipynb
```

Everything runs on CPU in a few minutes; there are no external downloads (the dataset ships with the repo).

---

## Skills demonstrated

`Machine Learning` · `Dimensionality Reduction (PCA, LDA/MDA)` · `Model Selection & Cross-Validation` · `SVM / Neural Networks / Discriminant Analysis` · `Statistical rigor (baselines, confidence intervals, leakage prevention)` · `Reproducible experiments` · `Python scientific stack (NumPy, pandas, scikit-learn, Matplotlib)` · `Technical writing (LaTeX)`

---

## Author

- **Giorgia Colucci** — Politecnico di Torino  
- [![LinkedIn](https://img.shields.io/badge/LinkedIn-Giorgia%20Colucci-0A66C2?logo=linkedin)](https://www.linkedin.com/in/giorgia-colucci)  
- GitHub: [@GiorgiaColucci](https://github.com/GiorgiaColucci)  
Machine Learning and Artificial Intelligence coursework project.

## License

Released under the MIT License. The dataset is © its original authors (P. Cortez & A. Silva, UCI ML Repository) and is used here for educational purposes.

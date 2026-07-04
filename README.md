# Predicting Student Performance — A Comparative Study of Dimensionality Reduction and Classification Methods

<p align="left">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white">
  <img alt="NumPy" src="https://img.shields.io/badge/NumPy-scientific-013243?logo=numpy&logoColor=white">
  <img alt="pandas" src="https://img.shields.io/badge/pandas-dataframes-150458?logo=pandas&logoColor=white">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-notebook-F37626?logo=jupyter&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
</p>

> End-to-end machine learning project that predicts a student's final-grade band from demographic, social and school-related features, and — more importantly — **quantifies how much the choice of feature representation and model actually matters** once you evaluate honestly.

Academic project for the *Machine Learning and Artificial Intelligence* course, reworked as a self-contained, reproducible case study. Every modelling choice is stated, justified, and tested against a baseline and against sampling noise, rather than reported as a single lucky accuracy number.

---

## TL;DR

- **Task.** Multiclass classification of students into **Low / Medium / High** final-grade bands (Portuguese-course subset of the UCI *Student Performance* dataset, 649 students, 33 raw attributes → 43 features after encoding).
- **Pipeline.** Clean encoding → standardization → three feature representations (**full space**, **PCA @ 95% variance**, **Fisher/MDA 2-D**) → five classifiers (**LDA, QDA, linear & RBF SVM, k-NN, MLP**).
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
- **Fisher Discriminant Analysis (FDA/MDA)** — two-class and multiclass cases, projection vs. classification (LDA), geometric interpretation; a custom `MultipleFisherDiscriminantAnalysis` implementation (`FisherDA.py`).
- **LDA & QDA** — derived as Bayesian classifiers under shared vs. per-class covariance; used to explain QDA's overfitting in high dimensions.
- **Support Vector Machines** — primal/dual formulation, support vectors, hard vs. soft margin, linear and RBF kernels, hyperparameter tuning via Grid Search + cross-validation, decision-boundary visualization in the LDA 2-D projection.
- **k-Nearest Neighbors** — used as a probe for the curse of dimensionality.
- **Multi-Layer Perceptron** — activation choice (ReLU), cost function, back-propagation, and hyperparameter selection via Grid Search + CV.
- **Robust evaluation** — class-balance handling, balanced accuracy / per-class metrics, stratified k-fold cross-validation with confidence intervals.

---

## Key results & takeaways

- On overlapping, noisy classes, **model complexity buys little**: linear boundaries (LDA, linear SVM) perform on par with the RBF SVM and the MLP.
- **QDA collapses** relative to LDA — estimating a full covariance per class on a few hundred samples in 43 dimensions is unstable, compounded by correlated one-hot features.
- **k-NN underperforms** because Euclidean distances concentrate in high dimensions (demonstrated quantitatively, not just asserted).
- The **cross-validated confidence intervals of the leading models overlap** → the differences seen on any single split are within sampling noise. The honest conclusion is stated as such.
- Feature representation matters more for *interpretability and stability* than for raw accuracy here: the compact supervised (MDA) space keeps performance while making the decision geometry visualizable.

---

## Repository structure

```
.
├── s323617_COLUCCI_Giorgia_student_completo.ipynb   # main notebook (theory + code + analysis)
├── FisherDA.py                                       # custom Fisher/MDA implementation (scikit-learn-style API)
├── student-por.csv                                   # dataset (Portuguese course)
├── latex_tesina/                                     # written thesis + slides (LaTeX)
│   ├── main.tex / main.pdf
│   └── presentazione_orale.tex / .pdf
└── README.md
```

> The notebook is the single source of truth: the LaTeX report and the slides are kept in sync with it.

---

## Getting started

```bash
# 1. clone
git clone https://github.com/<your-username>/student-performance-ml.git
cd student-performance-ml

# 2. environment
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy pandas scikit-learn matplotlib scipy jupyter

# 3. run
jupyter notebook s323617_COLUCCI_Giorgia_student_completo.ipynb
```

Everything runs on CPU in a few minutes; there are no external downloads (the dataset ships with the repo).

---

## Skills demonstrated

`Machine Learning` · `Dimensionality Reduction (PCA, LDA/MDA)` · `Model Selection & Cross-Validation` · `SVM / Neural Networks / Discriminant Analysis` · `Statistical rigor (baselines, confidence intervals, leakage prevention)` · `From-scratch algorithm implementation` · `Python scientific stack (NumPy, pandas, scikit-learn, Matplotlib)` · `Technical writing (LaTeX)`

---

## Author

**Giorgia Colucci** — Politecnico di Torino
Machine Learning and Artificial Intelligence coursework project.

## License

Released under the MIT License. The dataset is © its original authors (P. Cortez & A. Silva, UCI ML Repository) and is used here for educational purposes.

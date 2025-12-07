# COGS118A Final Project

**A Comparative Evaluation of Machine Learning Classifiers on Biomedical EEG and Voice Datasets**  
**Author:** Laurentia Liennart  
**Course:** COGS 118A — Supervised Machine Learning Algorithms  
University of California, San Diego

---

## **Overview**

This repository contains the implementation, analysis, and report for a comparative study evaluating four supervised learning algorithms across three biomedical datasets. The project examines how Logistic Regression, SVM with RBF kernel, Random Forest, and Multi-Layer Perceptron generalize across datasets with differing nonlinear structure, noise levels, and sample sizes.

Experiments were conducted in Python using scikit-learn, following an empirical evaluation design inspired by Caruana and Niculescu-Mizil (2006). Each classifier was tuned using grid search with 5-fold cross-validation, evaluated under multiple train/test splits, and repeated across three randomized trials to ensure robustness.

---

## **Repository Structure**

```
COGS118A_Final_Project/
│
├── notebooks/
│   ├── EEG_Eye_State.ipynb
│   ├── BEED_Epilepsy.ipynb
│   ├── Parkinsons.ipynb
│   └── Comparison_Analysis.ipynb
│
├── plots/
│   ├── eye_state_testacc_vs_trainsize.png
│   ├── eye_state_cv_vs_testacc.png
│   ├── beed_testacc_vs_trainsize.png
│   ├── beed_cv_vs_testacc.png
│   ├── parkinsons_testacc_vs_trainsize.png
│   ├── parkinsons_cv_vs_testacc.png
│   ├── combined_testacc_vs_trainsize.png
│   ├── combined_cv_vs_testacc.png
│   └── model_comparison_across_datasets.png
│
├── results/
│   ├── eeg_eye_results_summary.csv
│   ├── beed_results_summary.csv
│   ├── parkinsons_results_summary.csv
│
├── data/
│   └── README.md
│
├── report/
│   └── COGS118A_Final_Report.pdf
│
└── README.md
```

---

## **Datasets**

Raw datasets are not included. They can be downloaded from the UCI Machine Learning Repository:

* EEG Eye State: [https://archive.ics.uci.edu/dataset/264](https://archive.ics.uci.edu/dataset/264)
* BEED Epilepsy Detection: [https://archive.ics.uci.edu/dataset/1134](https://archive.ics.uci.edu/dataset/1134)
* Parkinson’s Disease Classification: [https://archive.ics.uci.edu/dataset/174](https://archive.ics.uci.edu/dataset/174)

---

## **Methods and Experimental Setup**

Four classifiers were trained and tuned:

* Logistic Regression
* Support Vector Machine (RBF Kernel)
* Random Forest Classifier
* Multi-Layer Perceptron (Neural Network)

All models were tuned using **5-fold cross-validation** within each training split.
Data partitions: **20/80**, **50/50**, **80/20**, each repeated across **three random seeds**.
Accuracy was used as the primary evaluation metric.

---

## **Summary of Mean Test Accuracies (Across All Trials & Splits)**

*These aggregated values are based on model rankings visible in the plots and summary tables.*

### **EEG Eye State Dataset**

| Model               | Mean Test Accuracy |
| ------------------- | ------------------ |
| Logistic Regression | ~0.63              |
| SVM (RBF Kernel)    | ~0.90–0.93         |
| Random Forest       | ~0.92–0.94         |
| MLP                 | ~0.88–0.92         |

---

### **BEED Epilepsy Detection Dataset**

| Model               | Mean Test Accuracy |
| ------------------- | ------------------ |
| Logistic Regression | ~0.90              |
| SVM (RBF Kernel)    | ~0.96–0.98         |
| Random Forest       | ~0.95–0.97         |
| MLP                 | ~0.94–0.96         |

---

### **Parkinson’s Voice Dataset**

| Model               | Mean Test Accuracy |
| ------------------- | ------------------ |
| Logistic Regression | ~0.83–0.85         |
| SVM (RBF Kernel)    | ~0.88–0.91         |
| Random Forest       | ~0.85–0.89         |
| MLP                 | ~0.80–0.87         |

---

### **Overall Cross-Dataset Model Ranking**

| Rank | Model               |
| ---- | ------------------- |
| 1    | SVM (RBF Kernel)    |
| 2    | Random Forest       |
| 3    | MLP                 |
| 4    | Logistic Regression |

---

## **Requirements**

The project uses the following Python packages:

```
Python 3.8+
numpy
pandas
scikit-learn
matplotlib
seaborn
ucimlrepo
```

To install dependencies:

```
pip install numpy pandas scikit-learn matplotlib seaborn ucimlrepo
```

If running in Google Colab, all dependencies are available by default except `ucimlrepo`, which can be installed via:

```
!pip install ucimlrepo
```

---

## **How to Reproduce the Experiments**

1. Download datasets and place them in the `data/` folder, or upload them manually when running the notebooks in Google Colab.
2. Open the desired notebook from the `notebooks/` directory.
3. Adjust file paths as needed or mount Google Drive.
4. Run all cells to perform preprocessing, hyperparameter tuning, training, and evaluation.
5. Generated plots and summaries will be saved to the `plots/` and `results/` directories.

---

## **Summary of Findings**

* Nonlinear models (SVM-RBF, Random Forest, MLP) outperform Logistic Regression across all tasks.
* Random Forest shows high stability across EEG datasets; SVM performs best on the Parkinson’s dataset.
* Training size significantly affects performance, especially for SVM and MLP.
* Cross-validation is a reliable predictor of test accuracy, except in small datasets such as Parkinson’s.

A complete discussion of these findings is provided in the report.

---

## **Final Report**

The full NeurIPS-style report is located in:

```
/report/COGS118A_Final_Report.pdf
```

It includes detailed methodology, results, plots, and interpretations.

---

## **References**

* Caruana, R., & Niculescu-Mizil, A. (2006). *An Empirical Comparison of Supervised Learning Algorithms.* ICML.
* Breiman, L. (2001). *Random Forests.* Machine Learning.
* Pedregosa et al. (2011). *Scikit-Learn: Machine Learning in Python.* JMLR.
* UCI Machine Learning Repository Datasets.

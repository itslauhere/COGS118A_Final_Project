# README — COGS118A Final Project
**A Comparative Evaluation of Machine Learning Classifiers on Biomedical EEG and Voice Datasets**  
**Author:** Laurentia Liennart  
**Course:** COGS 118A — Supervised Machine Learning Algorithms**  
University of California San Diego

---

## **Overview**

This repository contains the implementation, analysis, and final report for a comparative study of four supervised learning classifiers evaluated on three biomedical datasets from the UCI Machine Learning Repository. The project investigates how Logistic Regression, Support Vector Machine with an RBF kernel, Random Forest, and Multi-Layer Perceptron generalize across datasets that differ in dimensionality, noise characteristics, and nonlinear structure. The study emphasizes reproducible experimental design, rigorous hyperparameter tuning, and cross-dataset comparison, following an empirical methodology modeled after Caruana and Niculescu-Mizil’s work on classifier evaluation.

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
│   ├── pdcls_results_summary.csv
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

Three datasets from the UCI Machine Learning Repository were used in this study. Each represents a distinct biomedical classification problem:

1. **EEG Eye State** — A large EEG dataset containing nearly 15,000 samples labeled as eye-open or eye-closed.
2. **BEED Epilepsy Detection** — A mid-sized dataset of engineered statistical EEG features used to classify seizure activity.
3. **Parkinson’s Disease Voice Dataset** — A small voice dataset with acoustic features designed to detect Parkinsonian dysphonia.

Due to licensing and file size constraints, dataset files are not included in this repository. Users may download them directly from UCI and place them into the `data/` directory or upload them manually when running notebooks in Google Colab.

---

## **Methods**

This project evaluates four supervised learning algorithms—Logistic Regression, Support Vector Machine with RBF kernel, Random Forest, and Multi-Layer Perceptron—on each dataset using a consistent experimental framework. Each dataset was evaluated under three train/test partitions (20/80, 50/50, and 80/20) to study how classifier performance changes with varying amounts of training data. For every partition, each model underwent hyperparameter tuning using grid search with 5-fold cross-validation to identify the configuration achieving the highest validation accuracy.

To ensure robustness, all experiments were repeated across three randomized trials per train/test split. Performance was assessed using classification accuracy, and additional statistics such as cross-validation accuracy and training accuracy were recorded for further analysis. Summary files containing the best hyperparameters and averaged results are located in the `results/` directory, while all visualizations were generated through dedicated notebooks and saved in the `plots/` directory. These plots illustrate dataset-specific trends, cross-dataset comparisons, and model behavior across training sizes.

---

## **Summary of Findings**

Across datasets, nonlinear models consistently outperform Logistic Regression, demonstrating that biomedical features—such as EEG signals and dysphonia measures—often exhibit nonlinear relationships that cannot be captured by linear decision boundaries. On the EEG Eye State and BEED datasets, Support Vector Machines and Random Forests achieved the highest accuracies, often exceeding 0.90. MLP performed competitively but showed greater variability, particularly with smaller training sizes. Logistic Regression consistently produced the lowest accuracies, highlighting its limited capacity in modeling complex feature interactions.

The Parkinson’s dataset exhibited greater variance in both cross-validation and test performance due to its small sample size, but SVM remained the strongest overall model, followed by Random Forest and MLP. Learning curves across all datasets showed that performance generally improved with larger training sizes for flexible models such as SVM and MLP, whereas Logistic Regression exhibited minimal gains. Cross-validation aligned closely with test accuracy for the EEG datasets, reflecting their relatively large sample sizes, while the Parkinson’s dataset showed greater fold-to-fold variation. Overall, the results demonstrate that classifier performance is jointly determined by model flexibility, dataset size, and the underlying structure of the features.

---

## **Mean Test Accuracies (Aggregated Across Trials and Splits)**

### **EEG Eye State**

| Model               | Mean Accuracy |
| ------------------- | ------------- |
| Logistic Regression | ~0.63         |
| SVM (RBF Kernel)    | ~0.90–0.93    |
| Random Forest       | ~0.92–0.94    |
| MLP                 | ~0.88–0.92    |

### **BEED Epilepsy Detection**

| Model               | Mean Accuracy |
| ------------------- | ------------- |
| Logistic Regression | ~0.90         |
| SVM (RBF Kernel)    | ~0.96–0.98    |
| Random Forest       | ~0.95–0.97    |
| MLP                 | ~0.94–0.96    |

### **Parkinson’s Voice Dataset**

| Model               | Mean Accuracy |
| ------------------- | ------------- |
| Logistic Regression | ~0.83–0.85    |
| SVM (RBF Kernel)    | ~0.88–0.91    |
| Random Forest       | ~0.85–0.89    |
| MLP                 | ~0.80–0.87    |

**Overall ranking:**

1. SVM (RBF Kernel)
2. Random Forest
3. MLP
4. Logistic Regression

---

## **Requirements**

This project uses the following Python libraries:

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

If running in Google Colab:

```
!pip install ucimlrepo
```

---

## **Reproducing the Experiments**

1. Download the datasets from UCI and place them into the `data/` directory, or upload them manually if using Colab.
2. Open any of the notebooks in the `notebooks/` directory.
3. Adjust dataset paths or mount Google Drive as needed.
4. Execute the notebook to reproduce preprocessing, model training, hyperparameter tuning, and evaluation.
5. Plots and summary files will be generated automatically and saved in `plots/` and `results/`.

---

## **Final Report**

The complete report, formatted in the NeurIPS template, is available at:

```
/report/COGS118A_Final_Report.pdf
```

---

## **References**

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324  

Caruana, R., & Niculescu-Mizil, A. (2006). An empirical comparison of supervised learning algorithms. In *Proceedings of the 23rd International Conference on Machine Learning* (pp. 161–168). Association for Computing Machinery. https://doi.org/10.1145/1143844.1143865  

Dua, D., & Graff, C. (2019). *UCI Machine Learning Repository*. University of California, Irvine, School of Information and Computer Sciences. https://archive.ics.uci.edu/  

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.  

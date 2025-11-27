# Short Performance Report

**Project:** Question Pair Similarity (`question-pair-similarity-analysis.ipynb`)
**dataset:** `train.csv`

## 1. Overview
A condensed summary of the experiments, the models tried, the number of processed samples, and key performance numbers extracted from the notebook outputs.

## 2. Models used
- Logistic Regression (with GridSearchCV)
- Random Forest / Gradient Boosting (classical ensemble models)
- Support Vector Machine (SVM) — referenced in results table
- Siamese-style / sentence-pair model(s)
- Keras ANN (tuned with Keras Tuner)

## 3. Number of processed data
- Example sequence shape printed in the notebook: **343,692** rows (reported as `Seq shapes: (343692, 50)`).

## 4. Performance summary (representative results extracted from notebook)
> These values were parsed from printed outputs and result tables in the notebook. Use the notebook to verify exact experiment-to-metric mapping.

| Model / Experiment | Accuracy | Precision | Recall | F1-score | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression (baseline) | 0.68311 | 0.55889 | 0.67298 | 0.61065 | 0.75302 |
| Logistic Regression (GridSearch best) | — | — | — | **0.69306** (best f1) | — |
| Keras ANN (example tuner trial / final) | **0.74059** | — | — | — | — |
| Best reported model (example) | **0.82675** | 0.74517 | 0.80666 | 0.77470 | **0.90306** |
| Various experiments (examples) | 0.8761 / 0.8296 / 0.9006 / 0.8321 | — | — | — | — |

## 5. Short interpretation
- **ROC AUC** values (up to ~0.90) indicate some models can rank positive examples well.
- **F1-scores** are lower than AUCs in many experiments, suggesting class imbalance or threshold selection issues.
- Performance varies substantially across experiments — ensure consistent data splits and evaluation sets for fair comparisons.

## 6. Quick recommendations
1. Address class imbalance (class weights, resampling, PR-AUC as metric).
2. Use stronger text embeddings (sentence-transformers / transformer fine-tuning).
3. Tune decision thresholds on validation to maximize F1 if that is the target metric.
4. Use stratified cross-validation and report mean ± std for stable estimates.

---

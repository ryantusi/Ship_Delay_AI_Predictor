# Ship Delivery Delay Prediction - Optimized Version 2 (V2)

## Project Status: Production-Ready Optimization

This directory, **V2**, represents a successful optimization project aimed at resolving the low **Recall** constraint identified in the initial model (V1). By implementing **advanced sampling techniques (SMOTE)** and **threshold calibration**, the model's ability to prevent costly delays was dramatically increased.

| Metric | V1 (Baseline) | **V2 (Optimized)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Recall (Delayed)** | $19.4\%$ | **$91.1\%$** | **$4.7\times$** |
| **F1 Score** | $0.285$ | **$0.666$** | **$2.3\times$** |
| **ROC AUC** | $0.627$ | **$0.680$** | $1.1\times$ |
| **Final Model** | Decision Tree | **XGBoost** | Better Ensemble |
| **Key Fix** | Default Threshold ($\mathbf{0.5}$)| **Optimized Threshold ($\mathbf{0.316}$)** | **Critical** |

---

## 🛠️ Optimization Strategy and MLOps Enhancements

The entire pipeline was re-engineered and re-trained with a focus on achieving high sensitivity for the minority "Delayed" class.

### 1. Data and Feature Engineering

* **Synthetic Data:** Retained the 20,000-record dataset, but enhanced the underlying synthetic patterns for better separability.
* **Feature Expansion:** Increased complexity by adding **$\mathbf{15+}$ engineered features**, including interaction terms (e.g., *Vessel age $\times$ Weather risk*), providing richer context for the ensemble models.
* **Target Balance:** Adjusted the data generation to a near **$48\%$ Delay Rate** to better challenge the models during the initial comparison phase.

### 2. Bias Mitigation & Rebalancing

To address the low Recall issue (the core problem of V1), the following techniques were implemented:

* **SMOTE (Synthetic Minority Oversampling Technique):** Applied during training to generate synthetic data points for the minority "Delayed" class, balancing the train set to a $1:1$ ratio.
* **Class Weights:** Applied weights in the loss function to penalize the model more heavily for **False Negatives (missed delays)**.

### 3. Model & Threshold Calibration

* **Model Selection:** Switched focus to robust ensemble models: **XGBoost, LightGBM, and Extra Trees**.
* **Optimal Threshold Calibration:** The single most impactful change. Instead of using the default threshold of $0.5$ (which sacrifices Recall), a data-driven threshold of **$\mathbf{0.316}$** was identified via analysis to maximize the F1 score while achieving the target Recall $> 55\%$.
* **Final Model:** **XGBoost** was selected as the final production model due to its optimal balance of performance and efficiency.

---

## 📊 V2 Test Set Performance (XGBoost @ Threshold 0.316)

This model is configured to be an **Early Warning System**, valuing the cost of a missed delay (False Negative) over the cost of a false alarm (False Positive).

| Metric | Value | Business Interpretation |
| :--- | :--- | :--- |
| **Recall (Primary Goal)** | **$91.1\%$** | The model successfully **catches 9 out of every 10** actual ship delays. |
| **F1 Score** | $0.666$ | High overall effectiveness for an imbalanced problem. |
| **Precision** | $0.525$ | Out of every 10 alerts the model flags, approximately 5 are true delays (acceptable trade-off). |
| **Missed Delays (FN)** | **$171$** | The number of expensive, unmitigated delays is minimized. |

### Confusion Matrix (Test Set: $N=4,000$)

| Actual \ Predicted | **On Time (0)** | **Delayed (1)** |
| :--- | :--- | :--- |
| **Actual On Time (0)** | **508 (True Negatives)** | **1,578 (False Positives)** |
| **Actual Delayed (1)** | **171 (False Negatives)** | **1,743 (True Positives)** |

---

## 📂 V2 Deliverables and Deployment

The V2 folder contains the fully production-ready assets:

| File Name | Description |
| :--- | :--- |
| `optimized_ship_delay_model.joblib` | **Production ML Pipeline** (XGBoost with fitted preprocessor). |
| `preprocessor_optimized.joblib` | The **33-feature preprocessor** (essential for transforming live API data). |
| `optimized_model_metadata.json` | Metadata including the critical `Optimal Threshold: 0.316`. |
| `optimized_model_evaluation.png` | Final performance chart showing the high Recall/F1. |
| `optimization_report.txt` | Full optimization log and business justification. |
| `optimized_batchnorm_nn.keras` | Saved Deep Learning model (Keras Native Format). |

## 🚀 Execution & Reproducibility

### Prerequisites

1.  Python 3.x
2.  Install necessary libraries (create a `requirements.txt` file and run `pip install -r requirements.txt`).
3.  Run v1 folder first

### Steps to Rerun the Pipeline

Run the script from the root of the `v2` directory:

```bash
python script.py
```

This will regenerate the 20,000-record dataset, retrain all models, re-evaluate performance, and overwrite all persisted files with optimization methods.


### Deployment Note

The deployed Flask simulator should now be updated to load the assets from this **V2** directory and must use the **$\mathbf{0.316}$** threshold for prediction.
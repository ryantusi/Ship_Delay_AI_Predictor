
# 🚢 Ship Delivery Delay Prediction - AI Model Version 1 (V1)

## Project Overview

This repository contains **Version 1 (V1)** of the **Ship Delivery Delay Prediction Pipeline**, a comprehensive Machine Learning and Deep Learning solution designed to predict whether a maritime shipment will arrive **On Time** or **Delayed** (Binary Classification).

This version demonstrates the **complete Deep Learning + MLOps pipeline** from data generation to deployment persistence but highlights an initial performance constraint due to reliance on a default prediction threshold.

-----

## 📂 Repository Structure (AI\_Model/v1)

The directory structure is organized to reflect the outputs of the automated ML pipeline (`script.py`).

| File/Folder | Type | Description |
| :--- | :--- | :--- |
| `script.py` | Python Script | **The core execution script.** Runs the entire pipeline: Data Generation, EDA, Preprocessing, Model Training (ML + DL), Evaluation, and Persistence. |
| `ship_logistics_raw_data.csv` | Dataset | The **synthetically generated** dataset (20,000 records) used for training this model version. |
| **Model Persistence** | | |
| `ship_delay_ml_model.joblib` | Model File | **Final Trained Traditional ML Model** (Decision Tree pipeline with optimized hyperparameters). |
| `preprocessor.joblib` | Pipeline File | **Fitted Data Preprocessing Pipeline.** Contains the Standard Scaler and One-Hot Encoder necessary to transform new, raw input data. |
| `ship_delay_simple_nn.h5` | Model File | Trained Deep Learning Model 1 (Simple Dense Network). |
| `ship_delay_resnet.h5` | Model File | Trained Deep Learning Model 3 (Residual Network). |
| **Metadata & Reports** | | |
| `model_metadata.json` | JSON | Configuration and final metrics of the selected model for deployment verification. |
| `feature_info.json` | JSON | List of all 33 features and the final input dimension (64). |
| `comprehensive_project_report.txt` | Text | Full textual output summary of the entire pipeline run. |
| **Visualizations** | | |
| `eda_visualization.png` | Image | Key insights from the Exploratory Data Analysis. |
| `final_model_comprehensive_evaluation.png` | Image | Confusion Matrix, ROC Curve, and Feature Importance for the final Decision Tree model. |
| `ml_vs_dl_comparison.png` | Image | Comparison of AUC and F1-scores across all ML and DL models. |
| `dl_training_history.png` | Image | Loss and AUC curves for all Deep Learning models. |

-----

## 🚀 Execution & Reproducibility

### Prerequisites

1.  Python 3.x
2.  Install necessary libraries (create a `requirements.txt` file and run `pip install -r requirements.txt`).

### Steps to Rerun the Pipeline

Run the script from the root of the `v1` directory:

```bash
python script.py
```

This will regenerate the 20,000-record dataset, retrain all models, re-evaluate performance, and overwrite all persisted files.

-----

## 🎯 Model Performance & Architectural Insights

### 1\. Final Selected Model (V1)

| Metric | Selected Model | Value |
| :--- | :--- | :--- |
| **Model Type** | Decision Tree (Traditional ML) | N/A |
| **Test F1 Score** | Decision Tree | **0.2848 (Initial)** |
| **Test AUC** | Decision Tree | 0.6265 |
| **Recall (Delayed)** | Decision Tree | **0.1938** |
| **Accuracy** | Decision Tree | 0.6460 |

### 2\. Deep Learning Performance

The best Deep Learning model, **ResNet**, achieved an F1 score of **0.2628** and an AUC of **0.5930**.

### 3\. Critical Observation: The Low F1 Score

The low F1 score and Recall value demonstrate a **key MLOps lesson**:

  * **Diagnosis:** The model is suffering from a **poor prediction threshold** due to class imbalance (Delay Rate: $36.4\%$). The model is too conservative, classifying **$1173$ actual delays as "On Time" (False Negatives)**.
  * **Technical Root Cause:** The model defaults to a $0.5$ probability threshold for binary classification, which is inappropriate when the target class prevalence is $\mathbf{< 50\%}$.
  * **Solution (Implemented in V2):** This requires moving the prediction logic to probability space and lowering the threshold (e.g., to $\mathbf{0.35}$ or $\mathbf{0.40}$) to boost **Recall** and maximize the F1-Score.

-----

## 🔑 Key Architectural Takeaways

  * **Robust Pipeline:** Demonstrated end-to-end processing capable of handling complex feature engineering and integration of heterogeneous models (Scikit-learn + Keras).
  * **Data Generation:** The ability to simulate high-risk feature contributions (e.g., vessel age $>20$ years adds $15\%$ to delay probability).
  * **Model Depth:** Implemented advanced architectures (Residual Network, Batch Normalization) for comparative analysis.
  * **Dimensionality Reduction (PCA):** Showed that the initial $64$ preprocessed features could be reduced to **$41$ components**, retaining $95\%$ of the variance, without significant performance loss.
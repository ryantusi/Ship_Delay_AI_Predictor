# 🚢 Real-Time Ship Delay Prediction Simulator (V2)

This repository hosts the production-ready Flask application that serves the optimized Machine Learning pipeline (Model Version 2). The application provides a web interface to input key maritime features and receive a real-time prediction on whether a vessel will arrive **On Time** or **Delayed**.

This simulator showcases a successful MLOps optimization effort where the model's ability to correctly identify actual delays (**Recall**) was increased from $19.4\%$ to $91.1\%$.

## 🎯 Key Achievements (V2 Model)

The deployed model utilizes **XGBoost** and an optimized prediction threshold ($\mathbf{0.316}$) to achieve superior performance tailored for cost-sensitive logistics decisions.

| **Metric** | **V1 (Baseline)** | **V2 (Optimized)** | **Business Impact** |
| :--- | :--- | :--- | :--- |
| **Recall (Delayed)** | $19.4\%$ | **$91.1\%$** | **$4.7\times$ improvement in catching delays.** |
| **F1 Score** | $0.285$ | **$0.666$** | Highly reliable model for imbalanced data. |
| **Optimal Threshold** | $0.5$ (Default) | **$0.316$** | Calibrated to minimize costly missed delays (False Negatives). |
| **Primary Goal** | ACHIEVED | The model now reliably functions as an **Early Warning System**. |

## 📦 Application Structure

The Flask application is contained within this directory. The structure is designed for simplicity and easy deployment.

```

.
├── app.py                      # Main Flask application and routing
├── utils.py                    # Helper functions for model loading and feature engineering
├── templates/
│   └── index.html              # The web interface (HTML/CSS)
└── models/
    ├── optimized_ship_delay_model.joblib # Final XGBoost Model Pipeline
    ├── preprocessor_optimized.joblib # Final Fitted Data Preprocessor
    ├── optimized_model_metadata.json # Stores critical settings (e.g., Threshold: 0.316)
    ├── optimized_simple_nn.keras     # Optimized Deep Learning model 1
    └── optimized_batchnorm_nn.keras  # Optimized Deep Learning model 2

````

## ⚙️ How to Run Locally

### Prerequisites

You need Python 3.x and a few key libraries.

1. **Install Python Dependencies:** You must install Flask, Pandas, scikit-learn, and the required Deep Learning backend (TensorFlow).

```bash
   pip install flask pandas numpy scikit-learn joblib
   pip install tensorflow gunicorn # gunicorn is for production deployment
````

2.  **Ensure Model Files are Present:** Verify all files listed in the `models/` folder above are present.

### Execution

1.  Navigate to the directory containing `app.py`.

2.  Start the Flask server:

    ```bash
    python app.py
    ```

3.  Open your browser to the URL displayed in the terminal (typically `http://127.0.0.1:5000/`).

## 🚢 Model Inference Process

The user interface requires input for the 25 primary features. The application handles the prediction internally via a robust pipeline:

1.  **Input Collection:** `app.py` collects 25 raw inputs from the user.
2.  **Feature Engineering:** `utils.py` calculates the 8 engineered features (totaling 33 features).
3.  **Preprocessing:** `preprocessor_optimized.joblib` transforms the 33 features into a normalized 64-dimension vector (ready for the model).
4.  **Prediction:** The final XGBoost model (loaded from `optimized_ship_delay_model.joblib`) calculates the delay probability.
5.  **Threshold Application:** The probability is classified using the critical **$\mathbf{0.316}$ threshold** to maximize the chance of catching actual delays (high Recall).

## 💡 Using the Simulator

The simulator provides default low-risk values. To test the high-risk scenario and validate the model's Recall capability, try the following inputs:

| **Feature** | **High-Risk Value** | **Rationale (Triggers Delay)** |
| :--- | :--- | :--- |
| **vessel\_age\_years** | $\mathbf{25}$ | Old Vessel (High Risk) |
| **weather\_risk\_score** | $\mathbf{9.0}$ | Extreme Conditions |
| **sea\_state** | $\mathbf{Very\ Rough}$ | High Environmental Risk |
| **origin\_port\_congestion** | $\mathbf{8.5}$ | High Operational Risk |
| **dest\_port\_congestion** | $\mathbf{8.0}$ | High Operational Risk |
| **days\_since\_maintenance** | $\mathbf{350}$ | Maintenance Overdue |
| **cargo\_utilization\_pct** | $\mathbf{98}$ | Near-Max Capacity |
| **customs\_complexity\_score** | $\mathbf{9.0}$ | High Administrative Risk |


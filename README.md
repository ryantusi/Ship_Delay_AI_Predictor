# 🚢 AI-Powered Ship Delivery Delay Prediction

## Project Overview

**DeepPort Forecast** is an end-to-end Machine Learning Operations (MLOps) project designed to predict whether a maritime container shipment will arrive **On Time** or **Delayed** (Binary Classification).

The core value of this system is to function as a proactive **Early Warning System**, dramatically increasing the detection rate of potential delays (Recall) to minimize associated logistics costs and improve customer trust.

### Key Achievements

This repository contains the full source code for model development (V1 and V2) and the deployed Flask application (`/app`), demonstrating high proficiency in data science, MLOps, and full-stack integration.

| Metric | V1 (Initial Baseline) | **V2 (Optimized Production)** | Business Impact |
| :--- | :--- | :--- | :--- |
| **Recall (Delayed)** | $19.4\%$ | $91.1\%$ | $4.7\times$ **improvement in detecting delays.** |
| **F1 Score** | $0.285$ | $0.666$ | Highly reliable model for imbalanced data. |
| **Final Model** | Decision Tree | **XGBoost** | Best performance/speed tradeoff. |
| **Deployment** | Flask API with optimized $\mathbf{0.316}$ threshold. | | |

## 🏗️ Project Architecture

The repository is structured around the two core functions: **Model Training/Versioning** and **Production Deployment.**

```

.
├── AI_Model/                           # ML Training and Versioning History
│   ├── v1/                             # Initial Baseline Models (Low Recall)
|       └── ...
│   └── v2/                             # Optimized Production Models (High Recall)
|       └── ...
├── app/                                # Flask Web Application (Simulator)
│   └── ...
├── LICENSE
├── README.md                           # This file
├── ...
└── requirements.txt                    # Python dependencies

````

## 🧠 Model Development & Optimization (AI\_Model Directory)

The `AI_Model` folder documents the evolution from a baseline academic project (V1) to a robust production system (V2).

### V1: Baseline Failure Analysis

* **Goal:** Establish MLOps framework (ML, DL, Ensemble comparison).
* **Key Insight:** Low F1-Score was diagnosed as a **technical flaw** arising from the inappropriate use of the default $0.5$ prediction threshold on imbalanced data ($\approx 36\%$ delay rate).
* **Models:** Decision Tree, Random Forest, Simple NN, ResNet.

### V2: Recall Optimization Success

* **Methodology:** Implemented techniques focused on minimizing False Negatives (missed delays) to save costs.
* **Techniques Implemented:**
    1.  **Threshold Calibration:** Optimal threshold identified at $\mathbf{0.316}$.
    2.  **Oversampling:** Used **SMOTE** during training to balance class distribution.
    3.  **Model Upgrade:** Switched best performing model to **XGBoost** for its superior handling of complex data patterns.
    4.  **Feature Expansion:** Created $>15$ interaction and aggregated risk features.
* **Result:** Recall jumped from $19.4\%$ to $91.1\%$.

## 🚀 Deployment & Running the Simulator

The core prediction service is exposed via the Flask web application in the `/app` directory.

### Prerequisites

1.  Python 3.x
2.  Install all required libraries using the `requirements.txt` file.

    ```bash
    pip install -r requirements.txt
    ```

### Running the Web Simulator Locally

1.  Navigate to the `/app` directory:

    ```bash
    cd app
    ```

2.  Start the Flask server using Gunicorn (recommended for stable local testing):

    ```bash
    gunicorn app:app
    ```

3.  Access the application at `http://127.0.0.1:5000` (or the address specified by gunicorn).

### Simulator Inference Workflow

The front-end user experience is built around a complex MLOps workflow running behind the scenes:

1.  **User Input:** User submits 25 primary features.
2.  **Feature Engineering:** `utils.py` calculates 8 engineered features (e.g., `port_total_congestion`).
3.  **Preprocessing:** `preprocessor_optimized.joblib` applies standardization and One-Hot Encoding.
4.  **Prediction:** The V2 XGBoost model generates the delay probability.
5.  **Classification:** The result is classified as "Delayed" only if the probability $\mathbf{> 0.316}$.

## 💡 Future Work & Roadmap

1.  **Real-Time Data Integration:** Replace simulated data inputs with live data fetched via IMO number using external APIs (AIS, Port Authorities, Maritime Weather).
2.  **Full-Stack Deployment:** Integrate this prediction service as the backend for my planned **AI-Powered Vessel Tracker** and **Ship Chandlers E-commerce** application.
3.  **Advanced Explainability:** Implement SHAP or LIME to provide per-prediction justification on why a ship is likely to be delayed.

## ✍️ Author & Licensing

**Ryan Tusi** | [LinkedIn](https://www.linkedin.com/in/ryantusi/) | [Github](https://github.com/ryantusi/) | [Portfolio](https://ryantusi.netlify.app/)

**Licensing:**
This project is licensed under the **Creative Commons**. See the included `LICENSE` file for full details.

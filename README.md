# 🧱 Rubberized Concrete Strength Prediction using Machine Learning

## 📌 Overview

This repository presents a machine learning-based prediction model for estimating the **compressive strength of rubberized concrete**. The model is trained on a comprehensive dataset compiled from multiple published studies and is designed to support both research and practical mix design evaluation.

An interactive web application has also been developed to enable real-time prediction based on user-defined input parameters.

---

## ⚙️ Model Description

* **Algorithm**: Gradient Boosting Regressor

* **Input Features (9)**:

  * `wc` — Water-cement ratio
  * `CR` — Rubber content
  * `SR` — Sand ratio
  * `CC` — Cement content (kg/m³)
  * `CFA` — Fine aggregate (kg/m³)
  * `CCA` — Coarse aggregate (kg/m³)
  * `sfc` — Superplasticizer dosage
  * `CS` — Curing age (days)
  * `TC` — Temperature (°C)

* **Output**:

  * Compressive strength (MPa)

📌 Note: The model was trained on **unscaled data**, so no feature scaling is applied during prediction.

---

## 🚀 Deployment

A lightweight interactive application has been developed using Streamlit to allow users to input mix design parameters and obtain instant predictions.

---

## 📁 Repository Structure

```
rubber-concrete-ml/
│
├── app.py                              # Streamlit web application
├── gradient_boosting_model.joblib      # Trained ML model
├── standard_scaler.joblib              # Scaler (for reproducibility)
├── requirements.txt                    # Dependencies
├── README.md                           # Project documentation
```

---

## ▶️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rubber-concrete-ml.git
cd rubber-concrete-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run app.py
```

---

## 🌐 Web Application

The deployed application allows:

* Input of mix design parameters
* Real-time prediction of compressive strength
* Easy usability without coding knowledge

*(Add your deployment link here after publishing)*

---

## 📊 Dataset

The dataset consists of **1200+ rubberized concrete samples** collected from multiple peer-reviewed sources. It includes a wide range of mix compositions and experimental conditions to ensure model robustness.

---

## 🔁 Reproducibility

All files required to reproduce the results are provided:

* Dataset (if included)
* Trained model (`.joblib`)
* Source code
* Deployment script

---

## 📖 Citation

If you use this work, please cite:

```
(Add your paper citation here after publication)
```

---

## 👨‍🔬 Author

Md. Sazzadul Islam
MSc Researcher (Civil Engineering)
Research Focus: Sustainable Construction Materials & Machine Learning Applications

---

## ⚠️ Disclaimer

This model is intended for research and preliminary design purposes. Predictions should be validated with experimental testing for critical applications.

---

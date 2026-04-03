# 🏦 Fake Banknote Detection using Machine Learning

## 📌 Overview

This project implements a machine learning-based system to classify banknotes as **authentic** or **counterfeit** using statistical features extracted from note images.
It includes a trained model, a REST API built with FastAPI, and a complete pipeline for prediction.

---

## 🚀 Features

* Binary classification of banknotes (Real vs Fake)
* REST API using FastAPI
* JSON-based input/output
* Scalable backend architecture
* Clean and modular code structure

---

## 🧠 Machine Learning Approach

The model is trained on extracted statistical features:

* Variance
* Skewness
* Curtosis (Kurtosis)
* Entropy

### Models Used

* Logistic Regression
* Support Vector Classifier (SVC)

### Output

* `1` → Authentic Banknote
* `0` → Counterfeit Banknote

---

## 🛠️ Tech Stack

* Python
* FastAPI
* Scikit-learn
* Uvicorn
* Pandas / NumPy

---

## 📂 Project Structure

```
fake-banknote-detection-ml/
│
├── app.py                 # FastAPI application
├── Banknote.py           # Input schema (Pydantic model)
├── classifier.pkl        # Trained ML model
├── model.ipynb           # Model training notebook
├── Banknote_dataset.csv  # Dataset
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/fake-banknote-detection-ml.git
cd fake-banknote-detection-ml
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
uvicorn app:app --reload --port 5000
```

Open in browser:

```
http://127.0.0.1:5000/docs
```

---

## 📡 API Endpoint

### POST `/predict`

#### Request Body

```json
{
  "variance": 2.3,
  "skewness": 6.5,
  "curtosis": -1.2,
  "entropy": 0.45
}
```

#### Response

```json
{
  "prediction": "Authentic Banknote"
}
```

---

## 📊 Model Performance

* High accuracy on structured dataset
* SVC demonstrates strong classification performance
* Model evaluated using train-test split and validation techniques

---

## ⚠️ Limitations

* Uses pre-extracted features (not raw images)
* Dataset is relatively simple and clean
* Not fully representative of real-world counterfeit detection complexity

---

## 🔮 Future Improvements

* Integrate image-based input using Computer Vision
* Deploy API to cloud (Render / AWS)
* Build frontend UI for real-time predictions
* Add model explainability (SHAP / feature importance)

---

## 📄 License

This project is for educational and portfolio purposes.

---

## 👤 Author

Developed as part of a machine learning project focusing on real-world classification systems and API deployment.

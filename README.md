# 📊 Smart BI Assistant

## 🚀 Overview

Smart BI Assistant is a lightweight data analytics platform that enables non-technical users to upload datasets and automatically generate insights, visualizations, and machine learning results.

The system simplifies data analysis by combining:

* Automated Exploratory Data Analysis (EDA)
* Interactive Visualizations
* Machine Learning (Supervised & Unsupervised)
* Rule-based Insight Generation

---

## 🎯 Problem Statement

Non-technical users generate large amounts of structured data but struggle to extract meaningful insights using complex tools like Excel or Power BI. This leads to underutilized data and poor decision-making.

---

## 💡 Solution

Smart BI Assistant automates the entire data analysis pipeline:

1. Upload dataset (CSV)
2. Perform EDA (summary stats, missing values)
3. Clean and preprocess data
4. Generate visualizations
5. Apply machine learning:

   * Supervised Learning (prediction)
   * Unsupervised Learning (clustering)
6. Display insights in a simple dashboard

---

## 🧠 Machine Learning Approach

### 🔴 Supervised Learning

* Triggered when a target column is selected
* Models used:

  * Logistic Regression (for classification)
  * Linear Regression (for continuous prediction)
* Evaluation:

  * Accuracy
  * F1 Score

### 🔵 Unsupervised Learning

* Triggered when no target is selected
* Model:

  * K-Means Clustering
* Evaluation:

  * Silhouette Score
  * Inertia

---

## ⚙️ Tech Stack

| Component        | Technology          |
| ---------------- | ------------------- |
| Frontend         | Streamlit           |
| Backend          | FastAPI             |
| Data Processing  | pandas              |
| Machine Learning | scikit-learn        |
| Visualization    | Plotly / Matplotlib |

---

## 🧱 System Architecture

```
User (Streamlit UI)
        ↓
FastAPI Backend (API Layer)
        ↓
Pandas (EDA + Cleaning)
        ↓
ML Module (scikit-learn)
        ↓
Rule-Based Insights
        ↓
Dashboard Output
```

---

## 🔁 Workflow

1. Upload dataset via Streamlit
2. FastAPI processes and stores data
3. EDA performed using pandas
4. Data cleaned and preprocessed
5. Visualizations generated
6. ML model applied:

   * With target → Supervised
   * Without target → Clustering
7. Insights generated using rule-based logic
8. Results displayed in dashboard

---

## 📊 Dataset

We use a **credit card transaction dataset** with features like:

* transaction_amount
* category
* merchant
* time
* fraud_flag (target column)

This dataset supports:

* Spending analysis
* Fraud prediction
* Customer segmentation

---

## 📁 Project Structure

```
smart-bi-assistant/
│
├── backend/
│   ├── main.py
│   ├── ml_module.py
│   ├── eda.py
│   ├── insights.py
│   ├── state.py
│
├── frontend/
│   ├── app.py
│   ├── api_client.py
│
├── data/
│   └── sample.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ▶️ How to Run

### 1. Clone Repository

```
git clone <your-repo-url>
cd smart-bi-assistant
```

### 2. Create Virtual Environment

```
python -m venv venv
```

### 3. Activate Environment

```
# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 4. Install Dependencies

```
pip install -r requirements.txt
```

### 5. Run Backend

```
cd backend
uvicorn main:app --reload
```

### 6. Run Frontend

```
cd frontend
streamlit run app.py
```

---

## ⚠️ Limitations

* In-memory data storage (not scalable)
* No advanced NLP or chat interface
* Designed as a 5-day MVP prototype

---

## 🚀 Future Enhancements

* Advanced model selection
* Hyperparameter tuning
* Database integration
* Real-time analytics
* Natural language querying

---

## 👥 Team

* Backend Development
* ML Engineering
* Frontend Development
* Data Processing
* Integration & Documentation

---

## 🎯 Key Value

Smart BI Assistant bridges the gap between raw data and actionable insights by delivering a simple, automated, and explainable analytics pipeline.

---

## 📌 Note

This project is built as a **minimal viable product (MVP)** focusing on clarity, functionality, and explainability within a short development timeline.

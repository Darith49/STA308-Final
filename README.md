# 🎓 Course Recommendation System

**STA308 Final Project | Machine Learning & Statistical Computing**  
*Stack: Python 3.11+ • Streamlit • Scikit-Learn • Plotly*  
*Design System: [Anthropic Claude Warm Editorial](https://github.com/VoltAgent/awesome-design-md/blob/main/design-md/claude/DESIGN.md)*

---

## 💡 Brief Explanation of the App

The **Course Recommendation System** is an intelligent academic advising web application designed to help university students and advisors select elective courses. 

### What the App Does:
1. **Academic Aptitude Profiling:** Students enter their grades across 6 core foundation subjects (*Calculus, Statistics, Programming, English, Physics, Economics*) or upload a CSV file. The app standardizes their performance, projects it onto latent academic dimensions via Principal Component Analysis (PCA), and matches the student with an academic archetype (*Quantitative Specialist, Computational & Tech, Balanced Achiever, Socio-Economic*).
2. **Personalized Elective Ranking:** Using trained regularized Ridge regression models, the system predicts expected performance across university electives (*Econometrics, Data Mining, Operations Research, NLP, Financial Math, Biostatistics, Robotics, Corporate Finance*).
3. **Uncertainty Quantification:** Every predicted grade is accompanied by an approximate 95% prediction interval ($\hat{y} \pm 1.96 \cdot \hat{\sigma}_\epsilon$) so students know the confidence of each estimate.
4. **Cohort Benchmarking:** The app analyzes historical outcomes from the 15 most similar students in a 1,200-student benchmark cohort.
5. **Prerequisite Gating:** The system automatically checks course prerequisites (e.g., *Requires Calculus ≥ 60*), clearly displaying which electives are eligible vs. blocked.
6. **Transparent Explainability:** Every recommendation can be expanded to show exactly *why* it was recommended, breaking down the point impact of each subject grade in plain language.
7. **Privacy-First Design:** All grades entered are processed strictly in-memory and are never stored or logged.

---

## 🚀 How to Run the Application

Follow these simple steps to run the project locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/Darith49/STA308-Final-s.git
cd STA308-Final-s
```

### 2. Install Required Dependencies
Ensure you have Python 3.10+ installed. Install the dependencies via `pip`:
```bash
pip install -r requirements.txt
```

### 3. Launch the Web App
Run the Streamlit application:
```bash
streamlit run app/app.py
```
*(Alternatively, on Windows PowerShell run `.\run.ps1 app` or on Unix/Mac run `make app`)*

The application will automatically open in your default web browser at:
👉 **`http://localhost:8501`**

---

## 🧪 How to Run Automated Tests

To run the full suite of 14 unit and integration tests (validating inputs, intervals, monotonicity, prerequisites, and view syntax):
```bash
pytest -v
```
*(Or on Windows PowerShell: `.\run.ps1 test`, or via Makefile: `make test`)*

---

## 🔄 How to Retrain Offline Models (Optional)

The pre-trained models and precomputed statistics are already saved in the `models/` directory for instant inference. If you wish to retrain the models and regenerate synthetic cohort data:
```bash
python src/train.py
```
*(Or on Windows PowerShell: `.\run.ps1 train`, or via Makefile: `make train`)*

---

## 📑 Pages in the Application

- **🏠 Home:** Overview of system features, 3-step workflow guide, and privacy notices.
- **🎯 Get Recommendations:** Enter foundation grades, load sample student archetypes, upload a CSV, and view top-3 electives with radar charts, prediction intervals, and explanations.
- **📊 Explore Data:** Interactive exploration of grade distributions, correlation heatmaps, PCA factor loadings, scree plots, and elective catalog statistics.
- **📈 Model Performance:** 5-fold cross-validation benchmarking comparing baseline mean, student-GPA baseline, linear regression, Ridge regression, and Random Forest ($R^2$, RMSE, MAE, NDCG@3, catalog coverage, and fairness audits).
- **📖 Methodology & Ethics:** System architecture diagrams, mathematical formulations, assumptions, limitations, and ethical safeguards.

---

## 📂 Project Directory Structure

```
STA308-Final-s/
├── .streamlit/
│   └── config.toml                  # Streamlit theme & layout configuration
├── app/
│   ├── app.py                       # Main application entry point & router
│   ├── utils.py                     # Plotly chart generators, CSS, artifact loader
│   └── views/
│       ├── home_view.py             # Landing page & workflow overview
│       ├── recommend_view.py        # Core recommendation UI & inputs
│       ├── explore_view.py          # Precomputed dataset EDA & distributions
│       ├── performance_view.py      # Cross-validation benchmarks & fairness audit
│       └── methodology_view.py      # Mathematical derivations & ethics
├── data/
│   ├── raw_student_data.csv         # Benchmark training cohort (1,200 students)
│   ├── sample_profiles.json         # Preset student archetypes
│   └── sample_student_template.csv  # CSV upload template
├── models/
│   ├── elective_models/             # Serialized Ridge regressors per course
│   ├── eda_summary.json             # Precomputed distributions & PCA scree data
│   ├── imputation_models.joblib     # Pre-trained missing grade regressors
│   ├── kmeans.joblib                # Fitted clustering model & archetype labels
│   ├── metadata.json                # Lineage, training timestamp, versions
│   ├── metrics.json                 # 5-fold CV comparisons & ranking metrics
│   ├── neighbors.joblib             # Standardized cohort matrix for k-NN
│   ├── pca.joblib                   # 3-component PCA projection
│   ├── prereqs.json                 # Course prerequisite rules
│   ├── residual_stats.json          # Cross-validated residual standard deviations
│   └── scaler.joblib                # StandardScaler fit on training core grades
├── src/
│   ├── __init__.py
│   ├── recommend.py                 # Pure Python recommendation logic (zero UI dependencies)
│   └── train.py                     # Offline data generation & training pipeline
├── tests/
│   ├── __init__.py
│   ├── test_app_syntax.py           # Smoke tests for app views & chart builders
│   └── test_recommender.py          # Unit & integration test suite
├── .gitignore                       # Git ignore configuration
├── Makefile                         # GNU Make build targets
├── run.ps1                          # PowerShell execution script for Windows
├── requirements.txt                 # Pinned Python package dependencies
├── runtime.txt                      # Python runtime for cloud deployment
└── README.md                        # Documentation & quick start guide
```

---

## 📊 Model Evaluation Summary

| Model Architecture | Cross-Val RMSE | Cross-Val MAE | $R^2$ Score | Error Reduction vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Ridge Regression (Production)** | **3.74 pts** | **2.98 pts** | **0.864** | **-26.8%** |
| Linear Regression | 3.75 pts | 2.98 pts | 0.863 | -26.7% |
| Random Forest Regressor | 3.84 pts | 3.06 pts | 0.857 | -25.0% |
| Student GPA Baseline | 4.31 pts | 3.42 pts | 0.820 | -15.8% |
| Baseline Mean | 5.12 pts | 4.10 pts | 0.000 | 0.0% |

- **NDCG@3 Ranking Accuracy:** `0.842` (vs `0.582` for popularity baseline)
- **Precision@3:** `0.781`
- **Catalog Coverage:** `100.0%` of electives actively recommended

---

## 📄 License
Academic Course Project (STA308). MIT License.

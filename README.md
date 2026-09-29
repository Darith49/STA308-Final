# Course Recommendation System

STA308 Final Project - Machine Learning & Statistical Computing

## Overview

The Course Recommendation System is an academic advising web application designed to help university students select elective courses. Based on a student's grades across six core foundation subjects, the system:

- Profiles academic strengths using Principal Component Analysis (PCA) and K-Means clustering.
- Predicts expected grades across university electives using regularized Ridge regression models.
- Provides 95% prediction intervals to quantify uncertainty for each prediction.
- Validates course prerequisite rules and provides transparent score explanations.
- Benchmarks performance against similar historical student outcomes.

## Getting Started

### Prerequisites

- Python 3.10 or higher

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Darith49/STA308-Final.git
cd STA308-Final
pip install -r requirements.txt
```

### Run the Application

Start the Streamlit web application:

```bash
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`.

### Run Tests

Run the automated test suite:

```bash
pytest
```

## Retraining Models (Optional)

To regenerate synthetic cohort data and retrain the machine learning models:

```bash
python src/train.py
```

## Project Structure

- `app/` - Streamlit application entry point, page views, and charting utilities.
- `data/` - Benchmark cohort dataset and archetype profiles.
- `models/` - Serialized Ridge regression models, scalers, PCA weights, and metrics.
- `src/` - Pure Python inference engine (`recommend.py`) and training script (`train.py`).
- `tests/` - Unit tests for input validation, prediction intervals, and view syntax.

## License

MIT License

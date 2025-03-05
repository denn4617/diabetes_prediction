# Diabetes Prediction Model

This project involves building and evaluating machine learning models to predict whether individuals have diabetes based on diagnostic measurements. Two main models, Random Forest and XGBoost, are used to train on the diabetes dataset and predict outcomes.

## Features and Labels

- Features: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age.
- Labels: Outcome (0 or 1, where 1 indicates diabetes).

## Project Structure

```bash
project_root/
│
├── data/                     # Folder for datasets
│   ├── kaggle_diabetes_dataset/  # Subfolder for Kaggle diabetes dataset
│       ├── diabetes.csv      # Main dataset
│
├── src/                      # Source code
│   ├── data_preprocessing.py # Script for data cleaning and preparation
│   ├── data_exploring.py     # Script for initial data exploration
│   ├── model_training.py     # Script for training Random Forest model
│   ├── model_training_xgb.py # Script for training XGBoost model
│   ├── result_analysis.py    # Script to analyze and compare model results using OpenAI API
│
├── venv/                     # Virtual environment (optional)
│
├── .gitignore
├── .env                      # Environment file for storing sensitive keys(Optional)
├── requirements.txt          # Required libraries
└── README.md
```

## How to Use

- Execute the data preprocessing script first to clean and prepare your data.
- Run the data exploring script to get insights into the dataset.
- Use either of the model training scripts to build and save the model.
- Use the result analysis script to generate performance insights and comparisons between the two models using OpenAI API.

## Setup and Running Instructions

### 1. Setup Python Environment

To set up the project environment and install necessary packages, follow these steps:

- Ensure Python 3.x is installed on your system.
- Optional: Set up a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

- Install required packages:

```bash
pip install -r requirements.txt
```

### 2. Create and Setup .env File

- Create a .env file in the project root directory.
- Add your OpenAI API key:

```bash
OPENAI_API_KEY=your-api-key-here
```

- This key will be used by the result_analysis.py script.

### 3. Run the Scripts

- Data Preprocessing:

```bash
python src/data_preprocessing.py

```

- Data Exploration:

```bash
python src/data_exploring.py

```

- Model Training (Random Forest):

```bash
python src/model_training.py

```

Model Training (XGBoost):

```bash
python src/model_training_xgb.py

```

Result Analysis:

```bash
python src/result_analysis.py

```

# AI Shayak
A comprehensive AI governance platform that evaluates machine learning models for bias, fairness, and transparency using scikit-learn and fairlearn libraries.

<p align="center">
  <img src="public/Screenshot1.png" alt="Home"/><br>
  <img src="public/Screenshot2.png" alt="Evaluation"/>
</p>

## Usage
1. **Upload Model**: Upload your trained ML model file (.pkl or .joblib format)
2. **Upload Dataset**: Upload your test dataset (.csv format) with features and target variable
3. **Specify Sensitive Column**: Enter the column name that contains the sensitive attribute (e.g., 'sex', 'race', 'age_group')
4. **Run Analysis**: Click to start the evaluation process
5. **View Results**: Review the comprehensive report including:
   - Overall fairness and transparency scores
   - Detailed accuracy metrics
   - Group performance analysis
   - Bias detection results
   - Actionable recommendations

## Supported Model Types
- Scikit-learn models (LogisticRegression, RandomForest, etc.)
- Any model that implements `.predict()` and optionally `.predict_proba()` methods
- Models saved using joblib or pickle

## Dataset Requirements
- CSV format with headers
- Must include target variable column (typically named 'income' or similar)
- Must include sensitive attribute column for bias analysis
- Categorical variables should be properly encoded
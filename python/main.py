
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

print("Preparing Demo Files for AI-SAHAYAK")
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income'
]
data = pd.read_csv(url, header=None, names=columns, na_values=' ?', skipinitialspace=True)
data.dropna(inplace=True)
X = data.drop('income', axis=1)
y = data['income'].apply(lambda x: 1 if x == '>50K' else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
preprocessor = ColumnTransformer(transformers=[
    ('num', 'passthrough', numerical_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

print("\nTraining a Logistic Regression model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")
model_filename = 'model.pkl'
joblib.dump(model_pipeline, model_filename)
print(f"\nModel saved as '{model_filename}'")
test_data_filename = 'test_data.csv'
X_test['income'] = y_test
X_test.to_csv(test_data_filename, index=False)
print(f"Test data for evaluation saved as '{test_data_filename}'")
print("\n--- You are ready to test your platform! ---")
print(f"Upload '{model_filename}' and '{test_data_filename}'.")
print("In your platform, specify 'sex' or 'race' as the sensitive attribute.")
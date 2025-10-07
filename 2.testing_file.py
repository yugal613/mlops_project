# --- Setup and Imports ---
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import os

# --- 1. Load Data ---
# Ensure your dataset is in the correct path or directory.
df = pd.read_csv("D:/HeartDiseasePrediction/venv/Dataset/HeartDiseaseKaggledataset_part1.csv")

# Define target and features
X = df.drop('target', axis=1)
y = df['target']

# --- 2. Define Preprocessing Pipeline ---

numerical_features = ['age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak']
categorical_features = X.select_dtypes(include='object').columns.tolist()

# Create Preprocessing Transformer
preprocessor = ColumnTransformer(
    transformers=[
        # Scaling for numerical features
        ('num', StandardScaler(), numerical_features),
        # One-Hot Encoding for categorical features
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# --- 3. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 4. Model Training (with MLFlow integration placeholder) ---

# Initialize the Model
model = LogisticRegression(solver='liblinear', random_state=42)

# Create a full pipeline (preprocessor + model)
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# --- LOCAL MLFlow Tracking (You must run this part in your local environment) ---
import mlflow
with mlflow.start_run(run_name="Logistic_Regression_Heart_Disease"):
    mlflow.log_param("model_type", "LogisticRegression")
    
    full_pipeline.fit(X_train, y_train)
    y_pred = full_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(full_pipeline, "model_pipeline")
# ---------------------------------------------------------------------------------

# --- Train the final pipeline and Save ---
full_pipeline.fit(X_train, y_train)

# Create the 'models' directory and save the pipeline
#os.makedirs('D:/HeartDiseasePrediction/venv/models', exist_ok=True)
joblib.dump(full_pipeline, 'D:/HeartDiseasePrediction/venv/models/heart_disease_model.pkl')

# Save feature metadata for the API's input validation and structure
feature_metadata = {
    "all_features": X_train.columns.tolist()
}
joblib.dump(feature_metadata, 'D:/HeartDiseasePrediction/venv/models/feature_metadata.pkl')

print("Model training complete and saved to models/heart_disease_model.pkl")
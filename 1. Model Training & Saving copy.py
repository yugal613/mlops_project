import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from joblib import dump, load 

# MLflow Imports
import mlflow
import mlflow.sklearn

# Scikit-learn Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline 
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import os

# --- MLflow Setup ---
mlflow.set_experiment("Heart Disease Prediction Pipeline Comparison_v3")

# --- Data Loading ---
# WARNING: Absolute paths are poor practice. Use a file name like "data.csv" if running locally.
DATA_PATH = "D:/HeartDiseasePrediction/venv/Dataset/HeartDiseaseKaggledataset_part1.csv"
try:
    df = pd.read_csv(DATA_PATH)
    X = df.drop('target', axis=1)
    y = df['target']
except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}. Please ensure the path is correct.")
    exit()

# --- Define Preprocessing Pipeline (ColumnTransformer) ---
numerical_features = ['age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak']
categorical_features = X.select_dtypes(include='object').columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ],
    remainder='passthrough'
)

# --- Data Splitting ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Data split into Train and Test sets.")


# --- Training/Tracking Function (MODIFIED) ---
def trainPipeline_mlflow(model_class, model_params, X_train_raw, X_test_raw, y_train, y_test):
    """Creates a full pipeline, trains, and tracks results with MLflow."""
    
    model_name = model_class.__name__
    
    # --- FIX: Conditionally pass random_state ---
    final_params = model_params.copy()
    if 'random_state' in model_class().get_params():
        final_params['random_state'] = 42
    
    # 1. Create the Full Pipeline (Preprocessor + Model)
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model_class(**final_params))
    ])

    # Start an MLflow run for each model
    with mlflow.start_run(run_name=model_name) as run:
        print(f"\n--- Starting MLflow Run for {model_name} ---")

        # Log all parameters used
        mlflow.log_params(full_pipeline['classifier'].get_params())
        
        # 3. Train the Pipeline
        full_pipeline.fit(X_train_raw, y_train)
        y_pred = full_pipeline.predict(X_test_raw)
        
        # 4. Log Custom Metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_weighted_f1-score", report['weighted avg']['f1-score'])
        
        # --------------------------------------------------------
        # 5. LOG RICH ARTIFACTS (PLOTS AND REPORTS)
        # --------------------------------------------------------
        
        # 5a. Log Confusion Matrix Plot
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        # Passing the full pipeline to ConfusionMatrixDisplay handles the preprocessor step
        ConfusionMatrixDisplay.from_estimator(full_pipeline, X_test_raw, y_test, ax=ax)
        ax.set_title(f"Confusion Matrix ({model_name})")
        plt.tight_layout()
        
        plot_filename = f"confusion_matrix_{model_name}.png"
        plt.savefig(plot_filename)
        mlflow.log_artifact(plot_filename)
        plt.close(fig) # Close the figure to free memory

        # 5b. Log Classification Report (Text File)
        report_str = classification_report(y_test, y_pred)
        report_filename = f"classification_report_{model_name}.txt"
        with open(report_filename, "w") as f:
            f.write(report_str)
        mlflow.log_artifact(report_filename)
        
        # 6. Log Model Artifact
        mlflow.sklearn.log_model(full_pipeline, "model_pipeline_artifact")
        
        # 7. Print Report
        print("--- Classification Report on Test Set ---")
        print(report_str)
        print(f"MLflow Run ID: {run.info.run_id}")


# --- Run Models and Track ---
print("\n--- Starting Model Training and MLflow Tracking ---")

# Define models and parameters to run
models_to_run = [
    (LogisticRegression, {"solver": "liblinear"}),
    (SVC, {"probability": True}), 
    (KNeighborsClassifier, {"n_neighbors": 5}), 
    (DecisionTreeClassifier, {}),
    (RandomForestClassifier, {"n_estimators": 100}),
    (AdaBoostClassifier, {}),
    (GradientBoostingClassifier, {}),
]

for model_class, params in models_to_run:
    trainPipeline_mlflow(model_class, params, X_train, X_test, y_train, y_test)


# ---------------------------------------------------------------------------------
# --- Final Model Artifact Logging and Local Saving (CLEANED PATHS) ---
# ---------------------------------------------------------------------------------

# Assuming RandomForestClassifier is the best performer
FINAL_MODEL_CLASS = RandomForestClassifier
FINAL_MODEL_PARAMS = {"n_estimators": 120}

# Train the final production pipeline on the FULL dataset
final_full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', FINAL_MODEL_CLASS(**FINAL_MODEL_PARAMS, random_state=42)) 
])

# Train on all data
final_full_pipeline.fit(X, y)

# Save the final model and metadata locally (using relative paths 'models/')
#os.makedirs('models', exist_ok=True) # Ensure 'models' folder exists in the current directory
joblib.dump(final_full_pipeline, 'D:/HeartDiseasePrediction/venv/models/heart_disease_model.pkl') 
feature_metadata = {"all_features": X.columns.tolist()}
joblib.dump(feature_metadata, 'D:/HeartDiseasePrediction/venv/models/feature_metadata.pkl')
print("\n--- Local Saving Complete ---")
print("Final production pipeline saved to models/heart_disease_model.pkl")

# --- Log Final Production Model to MLFlow ---
with mlflow.start_run(run_name="Production_Model_Final_Save") as run:
    mlflow.sklearn.log_model(
        sk_model=final_full_pipeline,
        artifact_path="production_model",
        registered_model_name="HeartDiseasePredictor_Pipeline",
        metadata={"data_source": DATA_PATH}
    )
    mlflow.log_param("Final_Model_Type", FINAL_MODEL_CLASS.__name__)
    mlflow.log_params(FINAL_MODEL_PARAMS)
    print(f"Logged final production pipeline to MLflow. Run ID: {run.info.run_id}")
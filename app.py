import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np

# Function to load models
@st.cache_resource
def load_model(model_name):
    return joblib.load(f'model/{model_name}.pkl')

# Function to preprocess new data (must be consistent with training preprocessing)
def preprocess_data(df_input):
    df = df_input.copy()

    # Drop Loan_ID if present (assuming it's not a feature)
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)

    # Impute missing numerical values with the median (using training medians)
    # For this example, we'll use placeholder medians. In a real app, you'd save these from training.
    # For demonstration purposes, using static medians or re-calculating on input if suitable.
    # Better practice: save preprocessor (e.g, SimpleImputer) during training and load it here.

    # Placeholder medians/modes (replace with actual values from training data for production)
    # Example: median_loan_amount = 128.0, mode_gender = 'Male', etc.
    # For simplicity, we will impute with current data's median/mode *for this app.py generation*.
    # In a real scenario, these values would be loaded from saved preprocessors.

    # Numerical imputation
    for col in ['LoanAmount', 'Loan_Amount_Term', 'Credit_History']:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

    # Categorical imputation
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed']:
        if col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])

    # Handle 'Dependents' column specific issue if any (e.g., '3+' to 3)
    # The previous crash analysis revealed that '3+' should be kept for OHE.
    # Removed .astype(int) so it gets one-hot encoded correctly with other object columns.
    # The replace('3+', '3') was removed to generate Dependents_3+ during OHE.
    # If 'Dependents' exists, ensure it is of object type for correct OHE of '3+'
    if 'Dependents' in df.columns:
        # Ensure 'Dependents' is object type for correct OHE of '3+'
        df['Dependents'] = df['Dependents'].astype(str)

    # Identify categorical columns for one-hot encoding
    categorical_cols = df.select_dtypes(include='object').columns

    # Apply one-hot encoding to categorical features
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df

# Streamlit App Title
st.title("ML Model Deployment with Streamlit")
st.write("Upload a CSV file to get predictions and evaluate various ML classification models.")

# File Uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded CSV
    input_df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data (first 5 rows):")
    st.write(input_df.head())

    # Separate target variable if it exists in the uploaded file
    # Assuming 'Loan_Status' is the target and needs to be encoded for evaluation
    y_true = None
    if 'Loan_Status' in input_df.columns:
        le = LabelEncoder()
        y_true = le.fit_transform(input_df['Loan_Status'])
        df_for_prediction = input_df.drop('Loan_Status', axis=1)
    else:
        df_for_prediction = input_df.copy()

    # Preprocess the uploaded data
    processed_df = preprocess_data(df_for_prediction)

    # Corrected expected_columns to include one-hot encoded Dependents columns based on crash analysis
    expected_columns = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History',
        'Dependents_1', 'Dependents_2', 'Dependents_3+', # Updated based on crash analysis
        'Gender_Male', 'Married_Yes', 'Education_Not Graduate', 'Self_Employed_Yes',
        'Property_Area_Semiurban', 'Property_Area_Urban'
    ]

    # Add missing columns with 0 and remove extra columns
    missing_cols = set(expected_columns) - set(processed_df.columns)
    for c in missing_cols:
        processed_df[c] = 0
    extra_cols = set(processed_df.columns) - set(expected_columns)
    for c in extra_cols:
        processed_df = processed_df.drop(c, axis=1)

    processed_df = processed_df[expected_columns] # Ensure order

    st.subheader("Preprocessed Data (first 5 rows):")
    st.write(processed_df.head())

    # Model Selection
    model_options = {
        "Logistic Regression": "logistic_regression_model",
        "Decision Tree Classifier": "decision_tree_model",
        "K-Nearest Neighbor Classifier": "knn_model",
        "Naive Bayes Classifier": "naive_bayes_model",
        "Random Forest Classifier": "random_forest_model",
        "XGBoost Classifier": "xgboost_model"
    }
    selected_model_name = st.selectbox("Choose a Machine Learning Model:", list(model_options.keys()))

    if st.button("Run Prediction and Evaluation"):
        model_filename = model_options[selected_model_name]
        model = load_model(model_filename)

        st.subheader(f"Predictions and Evaluation for {selected_model_name}")

        # Make predictions
        y_pred = model.predict(processed_df)

        # Display predictions
        predictions_df = pd.DataFrame({'Predicted Loan Status': y_pred})
        st.write("Predictions on uploaded data (first 5):")
        st.write(predictions_df.head())

        # If target variable was in the uploaded file, perform evaluation
        if y_true is not None and len(y_true) == len(y_pred):
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            mcc = matthews_corrcoef(y_true, y_pred)

            try:
                y_pred_proba = model.predict_proba(processed_df)[:, 1]
                auc_score = roc_auc_score(y_true, y_pred_proba)
            except AttributeError: # Some models like Decision Trees might not have predict_proba for all configurations
                auc_score = "N/A (model does not support predict_proba)"
            except ValueError: # Handle cases where there's only one class in y_true, causing AUC error
                auc_score = "N/A (single class in target for AUC calculation)"

            evaluation_metrics = {
                "Metric": ["Accuracy", "AUC Score", "Precision", "Recall", "F1 Score", "MCC Score"],
                "Value": [accuracy, auc_score, precision, recall, f1, mcc]
            }
            metrics_df = pd.DataFrame(evaluation_metrics)

            st.write("--- Evaluation Metrics ---")
            st.dataframe(metrics_df.set_index('Metric'))
        else:
            st.warning("Target variable 'Loan_Status' not found or inconsistent in uploaded data. Cannot perform full evaluation.")

else:
    st.info("Please upload a CSV file to begin.")

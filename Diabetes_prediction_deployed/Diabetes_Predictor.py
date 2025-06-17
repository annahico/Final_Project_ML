import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_and_preprocess_data():
    df = pd.read_csv('dataset/diabetes.csv')

    df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

    return df


def handle_missing_values(df):
    df_copy = df.copy()

    # Columns to process
    cols_to_impute = ['Glucose', 'BloodPressure',
                      'SkinThickness', 'Insulin', 'BMI']

    # Replace 0 with nan ( it doesn't work with NaN)
    df_copy[cols_to_impute] = df_copy[cols_to_impute].replace(0, np.nan)

    # Define imputation strategies
    impute_strategy = {
        'Glucose': df_copy['Glucose'].mean(),
        'BloodPressure': df_copy['BloodPressure'].mean(),
        'SkinThickness': df_copy['SkinThickness'].median(),
        'Insulin': df_copy['Insulin'].median(),
        'BMI': df_copy['BMI'].median()
    }

    # Apply imputation
    for col, value in impute_strategy.items():
        df_copy[col] = df_copy[col].fillna(value)

    return df_copy


def train_model(X, y):
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    # Creating and training model
    classifier = RandomForestClassifier(n_estimators=20, random_state=0)
    classifier.fit(X_train, y_train)

    return classifier


def main():
    df = load_and_preprocess_data()
    df_processed = handle_missing_values(df)

    # Prepare features and target
    X = df_processed.drop(columns='Outcome')
    y = df_processed['Outcome']

    # Model training
    model = train_model(X, y)

    # Create directory if it doesn't exist
    os.makedirs('Diabetes_prediction_deployed', exist_ok=True)

    # Save model in the specified directory
    model_path = os.path.join(
        'Diabetes_prediction_deployed', 'diabetes_prediction_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model trained and saved successfully at {model_path}!")


if __name__ == "__main__":
    main()

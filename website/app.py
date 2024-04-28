import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
disease_df = pd.read_csv("framingham.csv")
disease_df.drop(['education'], inplace=True, axis=1)
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)
disease_df.dropna(axis=0, inplace=True)

# Prepare features and target
X = disease_df[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']]
y = disease_df['TenYearCHD']

# Explicitly set feature names
feature_names = X.columns.tolist()

# Normalize the features
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# Train the model
logreg = LogisticRegression()
logreg.fit(X_scaled, y)

# Save the trained model and scaler
with open('logistic_regression_model.pkl', 'wb') as model_file:
    pickle.dump(logreg, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Prediction function
def predict(input_data):
    # Load the trained model and scaler
    with open('logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Preprocess the input data
    input_data_scaled = scaler.transform([input_data])

    # Predict
    prediction = model.predict(input_data_scaled)

    return prediction

# Example usage
input_data = [50, 1, 50, 170, 200, 100]  # Example input data
prediction = predict(input_data)
print("Prediction:", prediction)

import pandas as pd
import joblib

# Load the trained model
model = joblib.load("models/dropout_model.pkl")

# Load dataset to test predictions
data = pd.read_csv("data/student_data.csv")

# Select features
X = data.drop("Target", axis=1)

# Get dropout probability
probabilities = model.predict_proba(X)[:, 1]

# Create risk levels
def get_risk_level(probability):
    if probability < 0.30:
        return "LOW"
    elif probability < 0.60:
        return "MEDIUM"
    else:
        return "HIGH"

# Add predictions to dataset
data["Dropout_Probability"] = probabilities
data["Risk_Level"] = data["Dropout_Probability"].apply(get_risk_level)

# Show sample results
print("\nSTUDENT DROPOUT RISK PREDICTION")
print("=" * 60)

print(
    data[["Target", "Dropout_Probability", "Risk_Level"]]
    .head(10)
)

# Save results
data.to_csv("data/student_risk_predictions.csv", index=False)

print("\nRisk predictions saved successfully!")
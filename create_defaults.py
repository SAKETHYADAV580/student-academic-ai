import pandas as pd
import joblib

# Load the original dataset
data = pd.read_csv("data/student_data.csv")

# Remove the target column
X = data.drop("Target", axis=1)

# Calculate median value for every feature
default_values = X.median().to_dict()

# Save default values
joblib.dump(default_values, "models/default_values.pkl")

print("Default values created successfully!")
print("Number of features:", len(default_values))
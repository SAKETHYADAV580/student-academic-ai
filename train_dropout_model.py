import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Convert Target into binary values
# Dropout = 1
# Graduate and Enrolled = 0
data["Dropout_Risk"] = data["Target"].apply(
    lambda x: 1 if x == "Dropout" else 0
)

# Features
X = data.drop(["Target", "Dropout_Risk"], axis=1)

# Target
y = data["Dropout_Risk"]

# Split data: 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create AI model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, predictions)

print("\nMODEL RESULTS")
print("=" * 40)
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save the trained model
joblib.dump(model, "models/dropout_model.pkl")

print("\nModel saved successfully!")
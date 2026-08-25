import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load dataset
data = pd.read_csv("data/student_data.csv")

# Create binary target
data["Dropout_Risk"] = (
    data["Target"] == "Dropout"
).astype(int)

# Features and target
X = data.drop(["Target", "Dropout_Risk"], axis=1)
y = data["Dropout_Risk"]

# Save feature names for future predictions
feature_names = X.columns.tolist()

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Final model
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

# Train model
model.fit(X_train, y_train)

# Predict on unseen test data
predictions = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("\nFINAL MODEL RESULTS")
print("=" * 50)
print(f"Accuracy:  {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall:    {recall * 100:.2f}%")
print(f"F1 Score:  {f1 * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save final model
joblib.dump(model, "models/final_dropout_model.pkl")

# Save feature names
joblib.dump(feature_names, "models/feature_names.pkl")

print("\nFinal model saved successfully!")
print("Feature names saved successfully!")
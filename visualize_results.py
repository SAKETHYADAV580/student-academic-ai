import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Load dataset
data = pd.read_csv("data/student_data.csv")

# ==============================
# 1. STUDENT OUTCOME DISTRIBUTION
# ==============================

outcome_counts = data["Target"].value_counts()

plt.figure(figsize=(8, 5))
outcome_counts.plot(kind="bar")
plt.title("Student Academic Outcome Distribution")
plt.xlabel("Academic Outcome")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("models/outcome_distribution.png")
plt.show()


# ==============================
# 2. FEATURE IMPORTANCE
# ==============================

# Load final model and feature names
model = joblib.load("models/final_dropout_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
).head(10)

plt.figure(figsize=(10, 6))
plt.barh(importance["Feature"], importance["Importance"])
plt.title("Top 10 Features Affecting Dropout Prediction")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("models/feature_importance.png")
plt.show()


# ==============================
# 3. CONFUSION MATRIX
# ==============================

# Create the same test split
data["Dropout_Risk"] = (
    data["Target"] == "Dropout"
).astype(int)

X = data.drop(["Target", "Dropout_Risk"], axis=1)
y = data["Dropout_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

predictions = model.predict(X_test)

cm = confusion_matrix(y_test, predictions)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Dropout", "Dropout"]
)

display.plot()
plt.title("Dropout Prediction Confusion Matrix")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
plt.show()

print("\nCharts created successfully!")
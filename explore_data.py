import pandas as pd

# Load the student dataset
data = pd.read_csv("data/student_data.csv")

# 1. Dataset size
print("DATASET SHAPE:")
print(data.shape)

# 2. Column names
print("\nCOLUMN NAMES:")
print(data.columns.tolist())

# 3. First 5 students
print("\nFIRST 5 ROWS:")
print(data.head())

# 4. Data information
print("\nDATA INFORMATION:")
data.info()

# 5. Missing values
print("\nMISSING VALUES:")
print(data.isnull().sum())

# 6. Student outcome counts
print("\nTARGET DISTRIBUTION:")
print(data["Target"].value_counts())
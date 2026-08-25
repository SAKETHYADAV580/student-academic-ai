from ucimlrepo import fetch_ucirepo

dataset = fetch_ucirepo(id=697)

X = dataset.data.features
y = dataset.data.targets

data = X.copy()
data["Target"] = y

data.to_csv("data/student_data.csv", index=False)

print("Dataset downloaded successfully!")
print("Dataset shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())
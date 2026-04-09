# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load your CSV
df = pd.read_csv("./datasets/car_efficiency_data.csv")


# Step 3: Prepare features (X) and target (y)
X = df[['Weight_KG', 'Horsepower', 'Cylinders', 'Aero_Score', 'MPG']]  # features
y = df['Is_Hybrid']  # target

# Step 4: Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale features (optional, but helps Naive Bayes)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)

# Step 7: Predict
y_pred = nb.predict(X_test_scaled)

# Step 8: Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
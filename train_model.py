import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# Load dataset
data = pd.read_csv("dataset.csv", header=None)

# Split features (landmarks) and labels
X = data.iloc[:, :-1]  # All columns except last (coordinates)
y = data.iloc[:, -1]   # Last column (labels)

# Encode labels to numerical values
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Classifier (Random Forest for good performance with structured data)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and label encoder
with open("hand_sign_model.pkl", "wb") as f:
    pickle.dump({"model": clf, "label_encoder": le}, f)

print("Model trained and saved as hand_sign_model.pkl")

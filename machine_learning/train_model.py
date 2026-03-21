# Aref
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os


#step1 load the squats dataset

df = pd.read_csv("squats_dataset.csv")

print(df.shape)
print(df.head(3))
print(df['label'].value_counts())

#step 2 seperate features from target y

feature_columns = [
    'right_elbow_right_shoulder_right_hip',
    'left_elbow_left_shoulder_left_hip',
    'right_knee_mid_hip_left_knee',
    'right_hip_right_knee_right_ankle',
    'left_hip_left_knee_left_ankle'
]

x = df[feature_columns].values
y = df['label'].values

print("X shape:", x.shape)
print("y shape:", y.shape)
print("y sample:", y[:5])

#step 3 
# bad -> 0
# good -> 1

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Encoded y sample:", y_encoded[:5])
print("Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

#step 4 train and test split 80% train 20% split
#random_state=42 means the shuffle is reproducible
# stratify=y_encoded keeps the good/bad ratio equal in both splits

X_train, X_test, y_train, y_test = train_test_split(
    x, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("Training rows:", X_train.shape[0])
print("Testing rows: ", X_test.shape[0])

#step 5 train model
model = RandomForestClassifier(n_estimators=200, random_state = 42)
model.fit(X_train, y_train)

print("model trained")

#step6 evaluate on test set

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)

#step 7 save matrix as image
os.makedirs("evaluation_results", exist_ok=True)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Bad', 'Good'],
            yticklabels=['Bad', 'Good'])
plt.title('Confusion Matrix — RepRight')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig("evaluation_results/confusion_matrix.png")
print("Saved confusion matrix to evaluation_results/")

#step 8 save model
joblib.dump(model, "squat_model.pkl")
joblib.dump(le,    "label_encoder.pkl")
print("Saved squat_model.pkl and label_encoder.pkl")




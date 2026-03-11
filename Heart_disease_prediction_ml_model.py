# Heart Disease Prediction using Logistic Regression----[Simple ML project for analysis and prediction]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


print("Heart Disease Prediction using Machine Learning\n")

# Dataset Reading##
df = pd.read_csv("heart_cleveland_upload.csv")

# Dataset Checking
print("Dataset shape:", df.shape)
print("\nMissing values in dataset:\n")
print(df.isnull().sum())

# removal of  duplicate rows
df.drop_duplicates(inplace=True)


# Data Visualization 

plt.figure()
sns.countplot(x='condition', data=df)
plt.title("Heart Disease Distribution")
plt.xlabel("Condition (0 = No disease, 1 = Disease)")
plt.ylabel("Number of Patients")
plt.show()


# age distribution
plt.figure()
sns.histplot(df['age'], bins=10)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()


# correlation between features
plt.figure()
sns.heatmap(df.corr(), annot=True)
plt.title("Feature Correlation Heatmap")
plt.show()


#MODEL BUILDING

# separating input features and target
X = df.drop('condition', axis=1)
y = df['condition']

# splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# training the LOGISTIC REGRESSION MODEL
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# making predictions
y_pred = model.predict(X_test_scaled)

# checking accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy*100, 2), "%")


#  Confusion Matrix 

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Value")
plt.ylabel("Actual Value")
plt.show()


#  Example Prediction 
sample = X_test_scaled[0].reshape(1, -1)
prediction = model.predict(sample)

print("\nSample Prediction Result:")

if prediction[0] == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease Detected")
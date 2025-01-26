import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Step 1: Load the collected data
keypoints_data = np.load('collected_data/keypoints.npy')
labels = np.load('collected_data/labels.npy')

# Step 2: Check the shape of the loaded data
print("Keypoints data shape:", keypoints_data.shape)
print("Labels shape:", labels.shape)

# Step 3: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(keypoints_data, labels, test_size=0.2, random_state=42)

# Step 4: Train a simple Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Step 5: Evaluate the model on the test data
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save the trained model for later use
joblib.dump(model, 'exercise_classifier.pkl')
print("Model saved as 'exercise_classifier.pkl'")

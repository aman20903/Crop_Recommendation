import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load and prepare data
crop = pd.read_csv("Crop_recommendation.csv")
X = crop.drop(['label'], axis=1)
y = crop['label'].map({
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
    'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15,
    'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19,
    'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use only StandardScaler (removing MinMaxScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = rfc.predict(X_test_scaled)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")

# Test with sample inputs
def test_prediction(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    scaled_features = scaler.transform(features)
    prediction = rfc.predict(scaled_features)
    return prediction[0]

# Save model and scaler
pickle.dump(rfc, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Test cases
test_cases = [
    [40, 50, 50, 40.0, 20, 7.0, 100],
    [100, 90, 100, 50.0, 90.0, 6.5, 202.0],
    [10, 10, 10, 15.0, 80.0, 4.5, 10.0]
]

for test in test_cases:
    result = test_prediction(*test)
    print(f"Input: {test}")
    print(f"Predicted crop number: {result}\n")
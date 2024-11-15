from flask import Flask, request, render_template
import numpy as np
import pickle
import pandas as pd

# Load model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Retrieve and convert form data to float
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosporus'])
        K = float(request.form['Potassium'])
        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        ph = float(request.form['Ph'])
        rainfall = float(request.form['Rainfall'])

        # Input validation with detailed messages
        validation_errors = []
        
        if not (0 <= ph <= 14):
            validation_errors.append("pH must be between 0 and 14")
        
        if temperature < -20 or temperature > 60:
            validation_errors.append("Temperature must be between -20°C and 60°C")
        
        if humidity < 0 or humidity > 100:
            validation_errors.append("Humidity must be between 0% and 100%")
        
        if rainfall < 0:
            validation_errors.append("Rainfall cannot be negative")
            
        if N < 0:
            validation_errors.append("Nitrogen content cannot be negative")
            
        if P < 0:
            validation_errors.append("Phosphorus content cannot be negative")
            
        if K < 0:
            validation_errors.append("Potassium content cannot be negative")

        # If there are any validation errors, return them
        if validation_errors:
            return render_template('index.html', result="Error: " + "; ".join(validation_errors))

        # Create feature array and scale it
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        scaled_features = scaler.transform(features)
        
        # Predict with the model
        prediction = model.predict(scaled_features)

        # Map prediction to crop name
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon",
            11: "Grapes", 12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil",
            16: "Blackgram", 17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
            20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        if prediction[0] in crop_dict:
            crop = crop_dict[prediction[0]]
            result = f"{crop} is the best crop to be cultivated right there."
            
            # Add input values to the result for verification
            
        else:
            result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
            
    except ValueError as ve:
        result = f"Error: Please ensure all inputs are valid numbers"
    except Exception as e:
        result = f"Error: An unexpected error occurred - {str(e)}"

    return render_template('index.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
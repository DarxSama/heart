from flask import Flask, request
import joblib
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the model
model = joblib.load("heart.pkl")

@app.route('/api/heart', methods=['POST'])
def heart():
    age = int(request.form.get('age')) 
    sex = str(request.form.get('sex')).lower()
    blood_pressure = int(request.form.get('blood_pressure'))
    cholestoral = int(request.form.get('cholestoral'))
    blood_sugar_120 = int(request.form.get('blood_sugar_120'))

    if sex == 'man':
        sex = 1
    elif sex == 'woman':
        sex = 0
    else:
        return {'error': 'Invalid value for sex'}, 400  
    

    blood_sugar_120 = 1 if blood_sugar_120 >= 120 else 0
    
    
    x = np.array([[age, sex, blood_pressure, cholestoral, blood_sugar_120]])

    
    prediction = model.predict(x)

    
    return {'heart': int(prediction[0])}, 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load saved model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except:
    model = None  # Fallback if model missing

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read all form values correctly (case-sensitive)
        age = int(request.form['age'])
        gender = 1 if request.form['gender'] == 'Male' else 0
        Polyuria = int(request.form['Polyuria'])
        Polydipsia = int(request.form['Polydipsia'])
        sudden_weight_loss = int(request.form['sudden_weight_loss'])
        weakness = int(request.form['weakness'])
        Polyphagia = int(request.form['Polyphagia'])
        Genital_thrush = int(request.form['Genital_thrush'])
        visual_blurring = int(request.form['visual_blurring'])
        Itching = int(request.form['Itching'])
        Irritability = int(request.form['Irritability'])
        delayed_healing = int(request.form['delayed_healing'])
        partial_paresis = int(request.form['partial_paresis'])
        muscle_stiffness = int(request.form['muscle_stiffness'])
        Alopecia = int(request.form['Alopecia'])
        Obesity = int(request.form['Obesity'])

        # Make input array
        input_data = np.array([[age, gender, Polyuria, Polydipsia, sudden_weight_loss,
                                weakness, Polyphagia, Genital_thrush, visual_blurring,
                                Itching, Irritability, delayed_healing, partial_paresis,
                                muscle_stiffness, Alopecia, Obesity]])

        # Predict
        if model:
            pred = model.predict(input_data)
            prediction = 'Positive' if pred[0] == 1 else 'Negative'
        else:
            prediction = 'Error: Model not found'

        return render_template('result.html', prediction=prediction)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)

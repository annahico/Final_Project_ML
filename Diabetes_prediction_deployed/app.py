import os
import pickle
import numpy as np
from flask import Flask, render_template, request

model_path = os.path.join(os.path.dirname(
    __file__), 'diabetes_prediction_model.pkl')
classifier = pickle.load(open(model_path, 'rb'))

dataset_path = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), 'dataset', 'diabetes.csv')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)

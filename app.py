import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.classification import *

app = Flask(__name__)

model = load_model('deploy_pps') 
columns = ['WeekDay', 'Period', 'NSW_Price', 'NSW_Demand', 'Vic_Price', 'VIC_Demand', 'Transfer']

@app.route('/')
def PPP():
    return render_template("PPP.html")

@app.route('/predict', methods=['POST'])
def predict():
    data_unseen = pd.DataFrame([np.array([x for x in request.form.values()])], columns=columns)
    prediction = predict_model(model, data=data_unseen)
    prediction = prediction.Label[0]
    return render_template('PPP.html', pred='Predicted Functioning Status of the Power-Plant : {}'.format(prediction))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    prediction = prediction.Label[0]
    output = prediction
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
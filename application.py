# Web server.
from flask import Flask, request, render_template

# Data manipulation.
import numpy as np
import pandas as pd

# File handling.
import os

# Predictions.
from src.pipeline.predict_pipeline import InputData, PredictPipeline


application = Flask(__name__)


app = application


# Home page.

@app.route('/')
def index():
    return render_template('index.html')


# Prediction page.

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        input_data = InputData(
            age=request.form.get('age'),
            sex=request.form.get('sex'),
            job=request.form.get('job'),
            housing=request.form.get('housing'),
            saving_accounts=request.form.get('saving_accounts'),
            checking_account=request.form.get('checking_account'),
            credit_amount=request.form.get('credit_amount'),
            duration=request.form.get('duration'),
            purpose=request.form.get('purpose')
        )

        input_df = input_data.get_input_data_df()
        print(input_df)
        print('\nBefore prediction.')

        predict_pipeline = PredictPipeline()
        print('\nMid prediction')
        prediction = predict_pipeline.predict(input_df)
        print('\nAfter prediction.')

        return prediction
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)


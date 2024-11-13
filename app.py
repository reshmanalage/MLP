from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

application = Flask(__name__)

app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            # Get input data from the form
            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                reading_score=float(request.form.get('writing_score')),
                writing_score=float(request.form.get('reading_score'))
            )

            # Prepare data for prediction
            pred_df = data.get_data_as_data_frame()
            logging.info(f"Prediction data frame: \n{pred_df}")

            # Run prediction pipeline
            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            # Return the results
            logging.info(f"Prediction results: {results[0]}")
            return render_template('home.html', results=results[0])

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return render_template('home.html', results="Error in prediction. Please check your input.")

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)

from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipelines.Prediction_Pipeline import CustomData, PredictPipeline

# Initialize the Flask application
application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('home.html')

# Route for handling the prediction form
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # If the user just navigated to the page, show them the empty form
        return render_template('form.html')
    else:
        try:
            
            def parse_input(field_name):
                raw_string = request.form.get(field_name)
                # Split by comma, strip extra spaces, and convert to float
                return [float(x.strip()) for x in raw_string.split(',')]

            # Build the dictionary by parsing the 7 values for every single feature
            user_7_day_data = {
                "PRECIPITATION": parse_input('PRECIPITATION'),
                "MAX_TEMP": parse_input('MAX_TEMP'),
                "MIN_TEMP": parse_input('MIN_TEMP'),
                "AVG_WIND_SPEED": parse_input('AVG_WIND_SPEED'),
                "YEAR": parse_input('YEAR'),
                "WIND_TEMP_RATIO": parse_input('WND_TEMP_RATIO'),
                "MONTH": parse_input('MONTH'),
                "SEASON": parse_input('SEASON'),
                "LAGGED_PRECIPITATION": parse_input('LAGGED_PRECIPITATION'),
                "LAGGED_AVG_WIND_SPEED": parse_input('LAGGED_AVG_WIND_SPEED'),
                "TEMP_RANGE": parse_input('TEMP_RANGE')
            }

            # Pass the parsed lists directly into our kwargs backpack
            data = CustomData(**user_7_day_data)
            
            # Convert to DataFrame
            pred_df = data.get_data_as_data_frame()
            print("\nIncoming User DataFrame Shape:", pred_df.shape)

            # Initialize the engine and make the prediction
            predict_pipeline = PredictPipeline()
            risk_level, raw_probability = predict_pipeline.predict(pred_df)

            # Send the result back to the HTML page
            return render_template('result.html', results=risk_level, probability=round(raw_probability, 4))

        except ValueError as ve:
            # If the user types a letter instead of a number, or doesn't provide exactly 7 days
            return render_template('home.html', error_message=f"Input Error: Please ensure you enter exactly 7 numbers separated by commas for each field. Details: {ve}")
        except Exception as e:
            return render_template('home.html', error_message=f"An error occurred: {e}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
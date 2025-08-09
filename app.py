from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json

app = Flask(__name__)
model = None
metrics = {}

MODEL_PATH = 'models/ridge_model.pkl'
METRICS_PATH = 'models/metrics.json'
DATA_PATH = 'data/house_data.csv'

def load_resources():
    global model, metrics
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH,'r') as f:
            metrics = json.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # parse inputs
        form = request.form
        try:
            area = float(form.get('area', 1500))
            bedrooms = int(form.get('bedrooms', 3))
            bathrooms = int(form.get('bathrooms', 2))
            age = float(form.get('age', 10))
            distance = float(form.get('distance_to_city', 5))
            garage = int(form.get('garage_spaces', 1))
            garden = int(form.get('has_garden', 0))
        except Exception as e:
            return f"Invalid input: {e}", 400

        X = pd.DataFrame([{
            'area': area, 'bedrooms': bedrooms, 'bathrooms': bathrooms,
            'age': age, 'distance_to_city': distance, 'garage_spaces': garage,
            'has_garden': garden
        }])

        if model is None:
            return "Model not found. Please run train.py first.", 500

        pred = model.predict(X)[0]
        pred = float(pred)

        # Create a small actual vs predicted plot using a sample from data
        if os.path.exists(DATA_PATH):
            df = pd.read_csv(DATA_PATH)
            sample = df.sample(50, random_state=1)
            Xs = sample.drop(columns=['price'])
            ys = sample['price'].values
            y_preds = model.predict(Xs)

            plt.figure(figsize=(6,4))
            plt.scatter(ys, y_preds)
            plt.plot([ys.min(), ys.max()], [ys.min(), ys.max()], linewidth=2)
            plt.xlabel('Actual Price')
            plt.ylabel('Predicted Price')
            plt.title('Actual vs Predicted (sample)')
            plot_path = os.path.join('static', 'imgs')
            os.makedirs(plot_path, exist_ok=True)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_path, 'actual_vs_pred.png'))
            plt.close()
        else:
            y_preds = []

        return render_template('result.html', pred=round(pred,2), metrics=metrics)
    else:
        # Render form
        return render_template('index.html', metrics=metrics)

if __name__ == '__main__':
    load_resources()
    app.run(host='0.0.0.0', port=5000, debug=True)

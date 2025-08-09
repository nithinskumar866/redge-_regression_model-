import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
np.random.seed(42)

N = 800
# Features: area (sqft), bedrooms, bathrooms, age (years), distance_to_city(km), garage_spaces, has_garden(0/1)
area = np.random.normal(1500, 400, N).clip(300, 5000)
bedrooms = np.random.choice([1,2,3,4,5], size=N, p=[0.05,0.25,0.4,0.2,0.1])
bathrooms = (bedrooms - 1) + np.random.choice([1,0], size=N, p=[0.7,0.3])
age = np.random.exponential(20, N).clip(0,100)
distance = np.random.exponential(5, N).clip(0,50)
garage = np.random.choice([0,1,2], size=N, p=[0.3,0.5,0.2])
garden = np.random.choice([0,1], size=N, p=[0.6,0.4])

# True underlying linear relationship (made-up coefficients)
price = (
    area * 120
    + bedrooms * 10000
    + bathrooms * 7000
    - age * 200
    - distance * 3000
    + garage * 8000
    + garden * 15000
    + np.random.normal(0, 30000, N)  # noise
)

df = pd.DataFrame({
    'area': area.round(0).astype(int),
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'age': age.round(1),
    'distance_to_city': distance.round(2),
    'garage_spaces': garage,
    'has_garden': garden,
    'price': price.round(2)
})

df.to_csv('data/house_data.csv', index=False)

# Train/test split
X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)

print(f"Test RMSE: {rmse:.2f}")
print(f"Test R2: {r2:.4f}")

joblib.dump(model, 'models/ridge_model.pkl')

# Save a small JSON with metrics to show in the app
import json
with open('models/metrics.json', 'w') as f:
    json.dump({'rmse': float(rmse), 'r2': float(r2)}, f)
print('Saved model and dataset.')

# 🏠 House Price Prediction using Ridge Regression (Flask App)

## 📌 Project Overview
This project predicts **house prices** based on property and location features using a Ridge Regression model.  
It is deployed as an interactive **Flask web application** where users can input house details and get an estimated price instantly.

---

## 📂 Project Structure
```
house-price-ridge/
│
├── app.py                  # Flask application
├── train.py                # Model training script
├── models/
│   ├── ridge_model.pkl     # Trained Ridge Regression model
│   └── metrics.json        # Model performance metrics
├── data/
│   └── house_data.csv      # Dataset used for training/testing
├── templates/
│   ├── index.html          # Input form page
│   └── result.html         # Prediction result page
├── static/
│   ├── css/style.css       # App styling
│   └── imgs/               # Generated plots
└── README.md               # Project documentation
```

---

## ⚙️ How It Works  

### **Data Source**  
The model uses a **synthetic housing dataset** (`house_data.csv`) containing property attributes and sale prices.

---

### **Feature Selection**  
The following features are used as inputs for prediction:

- `area` → House area in square feet  
- `bedrooms` → Number of bedrooms  
- `bathrooms` → Number of bathrooms  
- `age` → Age of the property (in years)  
- `distance_to_city` → Distance from city center (in km)  
- `garage_spaces` → Number of garage spaces  
- `has_garden` → Whether the house has a garden (1 = Yes, 0 = No)  

---

### **Model**  
A **Ridge Regression** model is trained on the selected features.  
The Ridge model helps reduce overfitting by penalizing large coefficients.

---

### **Deployment**  
The trained model is deployed via a **Flask** application.  
The web interface allows users to:

1. Input house details  
2. Get a predicted price in real time  
3. View an **Actual vs Predicted** scatter plot from a sample dataset

---

## 🔧 Installation  

### 1. Install Dependencies
```bash
pip install flask scikit-learn pandas numpy matplotlib joblib
```

### 2. Train the Model (if not already trained)
```bash
python train.py
```

### 3. Run the Flask App
```bash
python app.py
```

Visit: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🖥️ Sample UI  

**Input Form Example:**
| Feature               | Example Value |
|----------------------|--------------|
| Area (sq ft)         | 1500         |
| Bedrooms             | 3            |
| Bathrooms            | 2            |
| Age (years)          | 10           |
| Distance to City (km)| 5             |
| Garage Spaces        | 1             |
| Has Garden           | 1 (Yes)      |

**Output Example:**
```
Predicted Price: $245,000
```

**Sample Plot:**
_A scatter plot comparing actual vs predicted prices for a sample of 50 houses._

---

## 🙋‍♂️ Author  
Nithin.S  
[GitHub Profile](#)

---

## 📘 License  
This project is open source and free to use.

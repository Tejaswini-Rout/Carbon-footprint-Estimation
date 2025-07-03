# Carbon-footprint-Estimation
# 🌍 Carbon Footprint Estimation and Reduction using AI

A final-year major project aimed at predicting and reducing industrial carbon emissions using a GRU-based deep learning model and synthetic industrial IoT datasets.

---

## 📘 Project Description

This project leverages Artificial Intelligence, specifically Gated Recurrent Units (GRUs), to forecast the carbon footprint in industrial operations. The system simulates real-time data using synthetic generation techniques and suggests actionable strategies to optimize energy consumption and reduce emissions.

---

## 🧠 Key Features

- 📊 Synthetic data generation simulating industrial metrics (electricity, fuel, water usage, etc.)
- 🔍 Data preprocessing (standardization, heatmap correlation, one-hot encoding)
- 🧠 GRU-based deep learning model
- 📈 Forecasts carbon footprint and visualizes predictions
- 🧾 Generates sustainability recommendations in real time
- ✅ Performance metrics: MAE, RMSE, R² Score

---

## ⚙️ Technologies Used

- Python
- TensorFlow & Keras
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Google Colab (for development)

---

## 📂 Project Structure

```
carbon-footprint-ai/
├── main.py                      # Main Python file with training and prediction logic
├── factory_data.json            # Generated synthetic dataset
├── carbon_model_gru.keras       # Trained model file
├── scaler.pkl                   # Scaler used for prediction
├── actual_vs_predicted.png      # Output visualization
├── correlation_heatmap.png      # Feature correlation heatmap
└── README.md                    # This file
```

---

## 📈 Results

- **Mean Absolute Error (MAE):** ~261.97  
- **Root Mean Squared Error (RMSE):** ~305.20  
- **R² Score:** 0.8749

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python main.py
```

> Make sure `factory_data.json` is in the same folder.

---

## 🌱 Future Scope

- Real-time IoT integration for live monitoring  
- Expand to multi-plant datasets  
- Deploy via cloud APIs for industrial usage  

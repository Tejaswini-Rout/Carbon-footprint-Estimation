!pip install tensorflow scikit-learn joblib pandas numpy
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
%matplotlib inline
def generate_data():
 start_time = pd.Timestamp.now().floor('H')
 timestamps = pd.date_range(start=start_time, periods=2000, freq='H')
 np.random.seed(42)
 electricity_usage = np.random.uniform(3000, 8000, size=2000)
 fuel_consumption = np.random.uniform(1500, 5000, size=2000)
 production_output = np.random.uniform(50, 500, size=2000)
 carbon_intensity = np.random.uniform(200, 600, size=2000)
 water_usage = np.random.uniform(500, 1500, size=2000)
 temperature = np.random.uniform(5, 40, size=2000)
 locations = np.random.choice(['North', 'South', 'East', 'West'], size=2000)
 seasons = timestamps.month % 12 // 3 + 1
 days = timestamps.dayofweek
 carbon_footprint = (
 0.45 * electricity_usage +
 0.3 * fuel_consumption +
 0.08 * production_output +
 0.05 * carbon_intensity +
 0.05 * water_usage +
 0.03 * temperature +
 np.random.uniform(-600, 600, size=2000)
 )
 working_nature = []
 for e, f, p, w in zip(electricity_usage, fuel_consumption, production_output, water_usage):
 if e > 7000 and f > 4000:
 working_nature.append("High Load")
 elif p < 100 or w > 1200:
 working_nature.append("Low Efficiency")
 else:
 working_nature.append("Normal Operation")
 suggestions = []
 for e, f, p, c, w in zip(electricity_usage, fuel_consumption, production_output, carbon_intensity,
water_usage):
 if e > 7000:
 suggestions.append("Install solar panels or switch to energy-efficient lighting.")
 elif f > 4000:
 suggestions.append("Use alternative fuels or optimize combustion processes.")
 elif p < 100:
 suggestions.append("Optimize production scheduling to reduce energy waste.")
 elif c > 500:
 suggestions.append("Implement carbon capture technologies or improve air filtration.")
 elif w > 1200:
 suggestions.append("Recycle water within the process to minimize consumption.")
 else:
 suggestions.append("Implement a comprehensive energy management system.")
 energy_data = pd.DataFrame({
 'timestamp': timestamps,
 'electricity_usage': electricity_usage,
 'fuel_consumption': fuel_consumption,
 'production_output': production_output,
 'carbon_intensity': carbon_intensity,
 'water_usage': water_usage,
 'temperature': temperature,
 'season': seasons,
 'location': locations,
 'day_of_week': days,
 'carbon_footprint': carbon_footprint,
 'working_nature': working_nature,
 'suggestions': suggestions
 })
 energy_data.to_json("factory_data.json", orient='records', lines=True)
 plt.figure(figsize=(12, 6))
 energy_data[['electricity_usage', 'fuel_consumption', 'production_output', 'temperature']].hist(bins=30,
edgecolor='black')
 plt.tight_layout()
 plt.suptitle("Feature Distributions", fontsize=14)
 plt.savefig("feature_distributions_enhanced.png")
 plt.show()
 sns.pairplot(energy_data.drop(columns=['timestamp', 'suggestions', 'working_nature', 'location']),
diag_kind='kde')
 plt.savefig("pairplot_enhanced.png")
 plt.show()
def load_data():
 return pd.read_json("factory_data.json", lines=True)
def preprocess_data(data):
 data = data.copy()
 data = pd.get_dummies(data, columns=['location', 'season'], drop_first=True)
 features = [
 'electricity_usage', 'fuel_consumption', 'production_output',
 'carbon_intensity', 'water_usage', 'temperature', 'day_of_week'
 ] + [col for col in data.columns if 'location_' in col or 'season_' in col]
 X = data[features]
 y = data['carbon_footprint']
 scaler = StandardScaler()
 X_scaled = scaler.fit_transform(X)
 joblib.dump(scaler, "scaler.pkl")
 X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
 plt.figure(figsize=(10, 6))
 sns.heatmap(pd.DataFrame(X_scaled, columns=features).corr(), annot=True, cmap='coolwarm',
linewidths=0.5)
 plt.title("Feature Correlation Heatmap")
 plt.savefig("correlation_heatmap_enhanced.png")
 plt.show()
 return X_train, X_test, y_train, y_test, data
def train_model():
 generate_data()
 data = load_data()
 X_train, X_test, y_train, y_test, full_data = preprocess_data(data)
 X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
 X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
 model = Sequential([
 GRU(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
 GRU(32, activation='relu'),
 Dense(1)
 ])
 model.compile(optimizer='adam', loss='mse')
 model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
 model.save("carbon_model_gru.keras")
 return X_test, y_test, full_data
def make_predictions():
 data = load_data()
 X_train, X_test, y_train, y_test, full_data = preprocess_data(data)
 X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
 model = tf.keras.models.load_model("carbon_model_gru.keras")
 predictions = model.predict(X_test).flatten()
 plt.figure(figsize=(10, 5))
 plt.scatter(y_test, predictions, alpha=0.5, edgecolors='k')
 plt.xlabel("Actual Carbon Footprint")
 plt.ylabel("Predicted Carbon Footprint")
 plt.title("Actual vs. Predicted Carbon Footprint")
 plt.savefig("actual_vs_predicted_enhanced.png")
 plt.show()
 print(f"MAE={mean_absolute_error(y_test, predictions):.2f},
RMSE={np.sqrt(mean_squared_error(y_test, predictions)):.2f}, R²={r2_score(y_test, predictions):.4f}")
def predict_for_date(input_date: str):
 data = load_data()
 data['date'] = pd.to_datetime(data['timestamp']).dt.date
 sample = data[data['date'] == pd.to_datetime(input_date).date()]
 if sample.empty:
 print("No data for this date.")
 return
 _, _, _, _, full_data = preprocess_data(data)
 sample_processed = full_data[full_data['date'] == pd.to_datetime(input_date).date()]
 features = sample_processed.drop(columns=['timestamp', 'carbon_footprint', 'working_nature',
'suggestions', 'date'])
 scaler = joblib.load("scaler.pkl")
 X_input = scaler.transform(features)
 X_input = X_input.reshape((X_input.shape[0], X_input.shape[1], 1))
 model = tf.keras.models.load_model("carbon_model_gru.keras")
 predictions = model.predict(X_input).flatten()
 print(f"Predictions for {input_date}:")
 for i, pred in enumerate(predictions):
 print(f" Record {i+1}: {pred:.2f} units CO₂")
if __name__ == "__main__":
 train_model()
 make_predictions()
 predict_for_date("2025-06-15") 

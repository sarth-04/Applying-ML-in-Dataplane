import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Simulated Data for Network Performance
np.random.seed(42)
data_size = 1000
data = pd.DataFrame({
    'feature1': np.random.randn(data_size),
    'feature2': np.random.randn(data_size),
    'feature3': np.random.randn(data_size),
    'network_latency': np.random.randn(data_size) * 100 + 200  # in milliseconds
})

# Splitting the data into train and test sets
X = data[['feature1', 'feature2', 'feature3']]
y = data['network_latency']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Designing and deploying a ML model (Random Forest Regressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictive analytics for forecasting network performance
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Visualizing the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Network Latency')
plt.ylabel('Predicted Network Latency')
plt.title('Actual vs Predicted Network Latency')
plt.show()

# Simulated increase in data throughput and operational efficiency by 25%
initial_throughput = 1000  # hypothetical initial throughput
improved_throughput = initial_throughput * 1.25
print(f"Initial Throughput: {initial_throughput} units")
print(f"Improved Throughput: {improved_throughput} units")

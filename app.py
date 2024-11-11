import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Create a Flask app
app = Flask(__name__)

# Load the dataset from the provided file path
data = pd.read_csv(r"C:\Users\91998\OneDrive\Desktop\Global_Superstore2.csv", encoding='ISO-8859-1')

# Convert 'Order Date' to datetime format with dayfirst=True to avoid warning
data['Order Date'] = pd.to_datetime(data['Order Date'], dayfirst=True)

# Set the 'Order Date' column as the index
data.set_index('Order Date', inplace=True)

# Visualize monthly sales trends
@app.route('/')
def home():
    # Aggregate sales data by month
    monthly_sales = data['Sales'].resample('ME').sum()  # Use 'ME' instead of 'M'
    
    # Plot the monthly sales data
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_sales, label='Monthly Sales')
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Monthly Sales Over Time')
    plt.legend()
    plt.savefig('static/sales_trends.png')  # Save the plot as a static image
    plt.close()
    
    return render_template('index.html', plot_url='static/sales_trends.png')

# Generate the report when the button is clicked
@app.route('/generate_report', methods=['POST'])
def generate_report():
    # Aggregate sales data by month
    monthly_sales = data['Sales'].resample('ME').sum()

    # Split the data into train and test sets (80% training, 20% testing)
    split_point = int(len(monthly_sales) * 0.8)
    train = monthly_sales[:split_point]
    test = monthly_sales[split_point:]

    # Fit the Holt-Winters model
    hw_model = ExponentialSmoothing(train, trend='mul', seasonal='mul', seasonal_periods=12)
    hw_fit = hw_model.fit()

    # Make predictions
    forecast = hw_fit.forecast(len(test))

    # Calculate evaluation metrics
    mse = mean_squared_error(test, forecast)
    mae = mean_absolute_error(test, forecast)
    mape = (abs((test - forecast) / test).mean()) * 100
    accuracy = 100 - mape

    # Return the report
    return render_template('report.html', mse=mse, mae=mae, mape=mape, accuracy=accuracy, forecast=forecast)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

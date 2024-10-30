import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Set page configuration
st.set_page_config(
    page_title="Company growth rate Prediction",
    page_icon="https://th.bing.com/th?id=OIP.aC6e5nrwEQyWRAkJ-lctRgHaHk&w=247&h=252&c=8&rs=1&qlt=90&o=6&dpr=2&pid=3.1&rm=2"
)

# Title and description
st.title("Company growth rate Prediction Model")
st.markdown("Enter market performance, inflation trends, competitor growth rates, consumer spending habits, and the company’s advertising budget.")

# Load and prepare data

data = pd.read_csv(r"company_growth_dataset.csv")
 
# Prepare features and target'
X = data[['Market_Index_Performance','Inflation_Rate','Competitor_Growth_Rates','Consumer_Spending_Rate','Company_Advertising_Budget']]
y = data['Company_Growth_Rate']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Create input fields
st.subheader("Enter Details:")
row1_col1, row1_col2 =st.columns(2)
with row1_col1:
    Market_Index_Performance = st.number_input("Market_Index_Performance", min_value=-7, max_value=11, value=1)
    
with row1_col2:
    Inflation_Rate = st.number_input("Inflation_Rate", min_value=0, max_value=7, value=1)

row2_col1,row2_col2=st.columns(2)
with row2_col1:
    Competitor_Growth_Rates=st.number_input("Competitor_Growth_Rates",min_value=-5,max_value=14,value=1)

with row2_col2:
    Consumer_Spending_Rate = st.number_input("Consumer_Spending_Rate", min_value=-3, max_value=13, value=1)

row3_col1=st.columns(1)[0]
with row3_col1:
    Company_Advertising_Budget=st.number_input("Company_Advertising_Budget", min_value=8000, max_value=442273, value=50000)
    
# Add predict button

if st.button("Company growth rate Prediction"):
    input_data = pd.DataFrame(
    [[Market_Index_Performance, Inflation_Rate, Competitor_Growth_Rates, Consumer_Spending_Rate, Company_Advertising_Budget]],
    columns=X_train.columns  # Set the same columns as X_train
    )


    # # Prepare input
    # input_data = np.array([[Market_Index_Performance,Inflation_Rate,Competitor_Growth_Rates,Consumer_Spending_Rate,Company_Advertising_Budget]])
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    # Display prediction
    st.success(f"Company growth rate Predicted: ${prediction:,.2f}")
 
    # Show additional model information
    st.subheader("Model Information:")
    st.write(f"Market index Coefficient: {model.coef_[0]:.2f}")
    st.write(f"Inflation rate Coefficient: {model.coef_[1]:.2f}")
    st.write(f"Competitor growth rate Coefficient: {model.coef_[2]:.2f}")
    st.write(f"Consumer Spending rate Coefficient: {model.coef_[3]:.2f}")
    st.write(f"Company Advertising Budget Coefficient: {model.coef_[4]:.2f}")


# Generate predictions for the test data
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
r2 = r2_score(y_test, y_pred)

# Plotting original vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Original Data', marker='o', linestyle='-', color='blue')
plt.plot(y_pred, label='Predicted Data', marker='x', linestyle='--', color='red')
plt.title("Comparison between Original and Predicted Company Growth Rates")
plt.xlabel("Data Points")
plt.ylabel("Company Growth Rate")
plt.legend()
plt.grid(True)

# Display the plot in Streamlit
st.pyplot(plt)

# Generate random data for 30 days (for example)
num_days = 30
data = np.random.randn(num_days, 5)

# Create a DataFrame with appropriate column names
chart_data = pd.DataFrame(data, columns=['Market_Index_Performance',
                                          'Inflation_Rate',
                                          'Competitor_Growth_Rates',
                                          'Consumer_Spending_Rate',
                                          'Company_Advertising_Budget'])

# Optionally, add an index for better visualization (dates)
chart_data.index = pd.date_range(start='2024-01-01', periods=num_days)

# Create the bar chart
st.bar_chart(chart_data)

num_days = 30
chart_data = pd.DataFrame(np.random.randn(num_days, 5),
                          columns=['Market_Index_Performance',
                                   'Inflation_Rate',
                                   'Competitor_Growth_Rates',
                                   'Consumer_Spending_Rate',
                                   'Company_Advertising_Budget'])

# Optionally, you could add an index for better visualization (dates)
chart_data.index = pd.date_range(start='2024-01-01', periods=num_days)

# Create the area chart
st.line_chart(chart_data)

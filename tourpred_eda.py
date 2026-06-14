import joblib
import pandas as pd

model=  joblib.load('model.pkl')
x_scaler= joblib.load('x_scaler.pkl')
y_scaler= joblib.load('y_scaler.pkl')


country = int(input("Enter the country code: "))
category = int(input("Enter the category code: "))
visitors = int(input("Enter the number of visitors: "))
rating = float(input("Enter the rating: ")) 
accommodation = int(input("Enter the accommodation code: "))
input_data = [[country, category, visitors, rating, accommodation]]

input_data_scaled = x_scaler.transform(input_data)

predicted_scaled = model.predict(input_data_scaled)

predicted_revenue = y_scaler.inverse_transform(predicted_scaled)

print("Prediction revenue for the input data,",input_data,":", predicted_revenue) 
 
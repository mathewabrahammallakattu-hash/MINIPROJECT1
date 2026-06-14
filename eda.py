import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib


df = pd.read_csv('tourism_dataset.csv')
#print(df.shape())
#print(df.isnull().sum())
df = df.drop_duplicates()
#print(df.shape)


labelencoder = LabelEncoder()
df['country_encoded'] = labelencoder.fit_transform(df['Country'])
df['category_encoded'] = labelencoder.fit_transform(df['Category'])
df['accommodation_encoded'] = labelencoder.fit_transform(df['Accomedation'])


x= df[["country_encoded", "category_encoded", "Visitors", "Rating", "accommodation_encoded"]]
y= df[['Revenue']]

x_scaler = StandardScaler()
y_scaler = StandardScaler() 

x_scaled = x_scaler.fit_transform(x)
y_scaled = y_scaler.fit_transform(y)



x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(x_train, y_train)
print(" coef =",model.coef_)
print("intercept =",model.intercept_)   

y_pred = model.predict(x_test)
error = mean_squared_error(y_test, y_pred)
rms = np.sqrt(error)
print(rms)

joblib.dump(model, 'model.pkl')
joblib.dump(x_scaler, 'x_scaler.pkl')
joblib.dump(y_scaler, 'y_scaler.pkl')
print("Model and scalers saved successfully.")

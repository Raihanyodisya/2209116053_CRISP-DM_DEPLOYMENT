import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

url = "Data_Cleaned_Fix.csv"
df = pd.read_csv(url)

st.title("Restaurant Revenue Analysis")
st.subheader("Dataset")
st.write(df.head(5))

st.subheader('Exploratory Data Analysis (EDA)')

st.header("Basic Statistics - Cuisine_Type")
st.write(df['Cuisine_Type'].describe())

st.header("Value Counts")
st.write(df['Cuisine_Type'].value_counts())

st.header("Value Counts Visualization")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Cuisine_Type')
plt.xticks(rotation=45)
st.pyplot(plt)

st.write("Grafik di atas memvisualisasikan jumlah restoran yang menyajikan setiap jenis masakan dalam bentuk diagram batang. Setiap batang mewakili satu jenis masakan, dan tinggi batang menunjukkan jumlah restoran yang menyajikan masakan tersebut. Dengan grafik ini, kita dapat dengan mudah melihat perbandingan jumlah restoran antara jenis masakan yang berbeda. Misalnya, jika satu batang lebih tinggi dari yang lain, itu menunjukkan bahwa jenis masakan tersebut lebih umum di antara restoran dalam dataset.")

st.title('Number of Customers by Cuisine Type')

customers_by_cuisine = df.groupby('Cuisine_Type')['Number_of_Customers'].sum().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(data=customers_by_cuisine, x='Cuisine_Type', y='Number_of_Customers')
plt.xlabel('Cuisine Type')
plt.ylabel('Number of Customers')
plt.title('Number of Customers by Cuisine Type')
plt.xticks(rotation=45)
st.pyplot(plt)

st.write("Grafik ini menunjukkan jumlah total pelanggan yang datang ke restoran untuk setiap jenis masakan (Cuisine Type) dalam dataset. Setiap batang pada grafik mewakili satu jenis masakan, dan tinggi batang menunjukkan jumlah total pelanggan yang datang ke restoran yang menyajikan masakan tersebut.")
st.write("Dari grafik ini, kita dapat dengan jelas melihat perbandingan jumlah pelanggan yang lebih tinggi pada restauran yang menyajikan masakan Japanese. Ini menunjukkan bahwa jenis masakan tersebut lebih populer di antara pelanggan yang tercakup dalam dataset. Ini memberikan wawasan yang berguna bagi pemilik restoran atau pengelola untuk memahami preferensi pelanggan dan kinerja restoran mereka dalam menarik pengunjung.")

cuisine_distribution = df['Cuisine_Type'].value_counts()
number_of_cuisines = len(cuisine_distribution)

dominant_cuisine = cuisine_distribution.idxmax()
dominant_cuisine_count = cuisine_distribution.max()

st.write("Jumlah jenis masakan yang berbeda:", number_of_cuisines)
st.write("Jenis masakan yang mendominasi:", dominant_cuisine)
st.write("Jumlah restoran yang menyajikan jenis masakan tersebut:", dominant_cuisine_count)

st.write("Dari hasil analisis, terdapat 4 jenis masakan yang berbeda yang tersedia dalam dataset, yaitu 1. Italian, 2. Japanese, 3. Mexican, dan 4. American. Dari keempat jenis masakan tersebut, jenis masakan yang mendominasi adalah Japanese, dengan jumlah restoran yang menyajikan masakan tersebut sebanyak", dominant_cuisine_count)
st.write("Hal ini menunjukkan bahwa masakan Japanese merupakan pilihan yang populer di antara restoran yang tercakup dalam dataset.")

st.title('Box Plot of Numerical Variables by Cuisine Type')

numerical_cols = ['Number_of_Customers', 'Menu_Price', 'Marketing_Spend', 'Average_Customer_Spending', 'Promotions', 'Reviews', 'Monthly_Revenue']

for col in numerical_cols:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x='Cuisine_Type', y=col, ax=ax)
    ax.set_title(f'{col} by Cuisine Type')
    ax.set_xlabel('Cuisine Type')
    ax.set_ylabel(col)
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

file_path = 'gnb_model.pkl'
clf = joblib.load(file_path)

url = "Data_Cleaned_Fix.csv"
df = pd.read_csv(url)

number_of_customers = st.number_input('Number of Customers', value=0)  # Default value is 0
menu_price = st.number_input('Menu Price', value=0)  # Default value is 0
marketing_spend = st.number_input('Marketing Spend', value=0)  # Default value is 0
average_customer_spending = st.number_input('Average Customer Spending', value=0)  # Default value is 0
promotions = st.number_input('Promotions', value=0)  # Default value is 0
reviews = st.number_input('Reviews', value=0)  # Default value is 0
monthly_revenue = st.number_input('Monthly Revenue', value=0)  # Default value is 0

if st.button('Predict'):
    input_data = [[number_of_customers, menu_price, marketing_spend, average_customer_spending, promotions, reviews, monthly_revenue]]
    # Perform prediction
    result = clf.predict(input_data)

    if result.size > 0:
        if result[0] == 1:
            st.write('Hasil Prediksi Cuisine_Type: Italian')
        elif result[0] == 2:
            st.write('Hasil Prediksi Cuisine_Type: Japanese')
        elif result[0] == 3:
            st.write('Hasil Prediksi Cuisine_Type: Mexican')
        elif result[0] == 4:
            st.write('Hasil Prediksi Cuisine_Type: American')
        else:
            st.write('Hasil Prediksi Cuisine_Type tidak valid')
    else:
        st.write('Maaf, tidak dapat membuat prediksi. Mohon periksa kembali input Anda.')
import streamlit as st
from datetime import date
import numpy as np
import sklearn
import pickle
import pandas as pd 

st.set_page_config(page_title= "Retail Sales Prediction",
                   layout= "wide",
                   initial_sidebar_state='expanded')   

# user input options
class options:
    Store_type_values = ['A', 'B','C']
    Store_type_dict = {'A':1, 'B':2, 'C':3}
    
    Holiday_values = [True,False]
    Holiday_dict = {True:1, False:0}
    
    department_values = [ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 16, 17, 18,
       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
       36, 37, 38, 40, 41, 42, 44, 45, 46, 47, 48, 49, 51, 52, 54, 55, 56,
       58, 59, 60, 67, 71, 72, 74, 77, 78, 79, 80, 81, 82, 83, 85, 87, 90,
       91, 92, 93, 94, 95, 96, 97, 98, 99, 39, 50, 43, 65]
    size_values=[202505, 125833, 207499, 123737, 203742, 152513,  42988, 203750,
       118221,  93188, 128107, 103681,  39690, 151315, 126512, 112238,
       203819, 158114, 140167, 204184, 155083, 196321,  34875, 200898,
       119557, 202307,  93638,  39910, 120653, 206302, 219622,  57197,
       203007,  41062, 184109,  70713, 114533,  37392, 155078, 205863]


# Streamlit page custom design
page = st.sidebar.selectbox("Select", ["About","Weekly Sales Prediction", "Recommendation"], index=0)
if page=="About":
  st.title("Welcome to the Retail Weekly Sales Forecasting")
  if st.button("Overview"):
    st.write(''' The  Retail Weekly Sales Forecasting tool is a user-friendly
             web application built to predict the price from the given  data. 
             Leveraging the power of Streamlit and Machine Learning.''')
    st.header("Key Features")
    st.markdown(''' 
                - Data Understanding
                - Data Preprocessing
                - Exploratory Data Analysis 
                - Feature Engineering
                - Model Building & Evalution''')
    st.header("Technology List ")
    st.markdown(''' 
                - Python
                - Scikit-Learn
                - Numpy
                - Pandas
                - Plotly
                - Matplotlib
                - Seaborn
                - Streamlit''')
# Get input data from users for prediction 
if page=="Weekly Sales Prediction":
     st.title("Welcome to Weekly Sales Forecasting")
     st.header("Please fill the below form")
     col=st.columns((3,3),gap='medium')
     with col[0]:
         user_date = st.date_input(label='Date', min_value=date(2010, 2, 5),
                                          max_value=date(2013, 12, 31), value=date(2010, 2, 5))
         store = st.number_input(label='Store', min_value=1, max_value=45,
                                        value=1, step=1)
         Type = st.selectbox(label='Store Type',options=options.Store_type_values)
         dept = st.selectbox(label='Department',options=options.department_values)
         holiday = st.selectbox(label='Holiday',options=options.Holiday_values)
         temperature = st.number_input(label='Temperature(°F)', min_value=-10.0,
                                              max_value=110.0, value=-7.29)
         fuel_price = st.number_input(label='Fuel Price', max_value=10.0,
                                             value=2.47)
         size = st.selectbox(label='Size',options=options.size_values)
         
     with col[1]:
                markdown1 = st.number_input(label='MarkDown1', value=0.000)

                markdown2 = st.number_input(label='MarkDown2', value=0.000)

                markdown3 = st.number_input(label='MarkDown3', value=0.000)

                markdown4 = st.number_input(label='MarkDown4', value=0.22)

                markdown5 = st.number_input(label='MarkDown5', value=0.000)

                unemployment = st.number_input(label='Unemployment',
                                               max_value=20.0, value=3.68)
                cpi = st.number_input(label='CPI', min_value=100.0,
                                      max_value=250.0, value=126.06)
    # user entered the all input values and click the button          
     if st.button('Submit'):
         #input transforming using scalar method with markdown
         pickel_in_reg=open("x_scaled_data_afterskewness","rb")
         model_data_reg=pickle.load(pickel_in_reg)
         user_data_reg = np.array([[user_date.day, user_date.month, user_date.year,
                                    store,dept,options.Store_type_dict[Type],
                                    size, 
                               options.Holiday_dict[holiday],
                               temperature, 
                               fuel_price,
                               np.log(float(markdown1)), 
                               np.log(float(markdown2)), 
                               np.log(float(markdown3)),
                               np.log(float(markdown4)), 
                               np.log(float(markdown5)), 
                               cpi, unemployment]])
         user_data_scaled_reg=model_data_reg.transform(user_data_reg)
         # Input give to model and predict the price 
         pickel_in_reg_pred=open("model_XGB_regression_without skewness and outlier after scaled.pkl","rb")
         reg_model=pickle.load(pickel_in_reg_pred)
         y_p = reg_model.predict(user_data_scaled_reg)
         Weekly_Sales_price=np.exp(y_p[0]) 
         #input transforming using scalar method without markdown
         pickel_in_without_markdown=open("scaled_df_afterskewness_withoutmarkdown","rb")
         model_data_without_markdown=pickle.load(pickel_in_without_markdown)
         user_data_without_markdown = np.array([[user_date.day, user_date.month, user_date.year,
                                    store,dept,options.Store_type_dict[Type],
                                    size, 
                               options.Holiday_dict[holiday],
                               temperature, 
                               fuel_price, 
                               cpi, unemployment]])
         user_data_scaled_without_markdown=model_data_without_markdown.transform(user_data_without_markdown)
         # Input give to model and predict the price 
         pickel_in_without_markdown_pred=open("model_XGB_regression_without skewness and outlier after scaledwithoutmarkdown.pkl","rb")
         without_markdown_model=pickle.load(pickel_in_without_markdown_pred)
         y_p_without_markdown = without_markdown_model.predict(user_data_scaled_without_markdown)
         Weekly_Sales_price_without_markdown=np.exp(y_p_without_markdown[0])
         
         st.success(f"The predicted Weekly_Sales_price for the given data with markdown is: {Weekly_Sales_price}")
         st.success(f"The predicted Weekly_Sales_price for the given data without markdown is: {Weekly_Sales_price_without_markdown}")
         st.header("Weekly Sales Forecasting Recommendation")
         st.markdown(''' 
                - Investing in larger stores or expanding existing ones to potentially increase sales revenue.
                - The Fuel prices increase, consumer purchasing power may decrease, leading to 
                  potential changes in consumer behavior and spending patterns. Monitor fuel price fluctuations 
                  closely and adjust pricing strategies or promotional activities accordingly to mitigate any negative impact on sales.
                - Optimize markdown campaigns by targeting specific products or time periods to maximize their impact on sales 
                   while minimizing potential margin.
                - Changes in unemployment rates may affect consumer confidence and spending behavior. Monitor economic 
                  indicators closely and adjust marketing strategies or product offerings accordingly to adapt to changing 
                  consumer preferences during periods of high unemployment.
                - Develop targeted marketing campaigns or promotions during peak holiday seasons to capitalize on 
                 increased consumer spending and drive sales growth.''')
     
     
     
         
        
       
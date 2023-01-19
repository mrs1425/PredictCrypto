#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 22:24:57 2022

@author: Reza & Yves
"""

import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import pandas as pd

# It is mostly used for finding out the relationship between variables and forecasting.
from sklearn.linear_model import LinearRegression

# This computes a least-squares regression for two sets of measurements.
from scipy.stats import linregress

# Pyplot is a collection of functions in the popular visualization package Matplotlib. Its functions manipulate elements of a figure, such as creating a figure, creating a plotting area, plotting lines, adding plot labels, etc.
import matplotlib.pyplot as plt

# some characteristics like the width and color of the spines, the font size of the y-axis label, the absence of a grid, etc that make up matplotlib's default style.
plt.style.use('fivethirtyeight')


# The method of Support Vector Classification can be extended to solve regression problems. This method is called Support Vector Regression.
from sklearn.svm import SVR 

# for split our data into train and test sets where feature variables are given as input in the method.
from sklearn.model_selection import train_test_split 

# yfinance is a popular open source library developed by Ran Aroussi as a means to access the financial data available on Yahoo Finance.
import yfinance as yf
st.set_option('deprecation.showPyplotGlobalUse', False)

# In[]: Front-End

#Streamlit is an open source app framework in Python language.
#It helps us create web apps for data science and machine learning in a short time.
#It is compatible with major Python libraries such as:
#    scikit-learn, Keras, PyTorch, SymPy(latex), NumPy, pandas, Matplotlib etc.


st.write('''
         # CRYPTO Analyzer by Python
         **Reza & Yves**
         
         ''')    
         
#img = Image.open('/Users/mina/Desktop/Entreprise/Lusern/Courses/Programming/Projects/Final/forex/Learning/Test/Stockpic.jpeg')         
#st.image(img,width=600,caption='Important: After Your Technical & Fundamental Analysis look our Predict')

with st.sidebar:
    selected = option_menu("Guide", ["Automatic-guide", 'Manually-guide','Project-Report'],default_index=1, 
        icons=['list-task', 'list-task','download'], menu_icon="cast")
    
    if selected == "Automatic-guide":
        col1, col2 = st.columns([0.8,0.2])
        with col1:               # To display the header text using css style
            st.markdown(""" <style> .font {
            font-size:20px ; font-family: 'Cooper Black'; color: #FF9633;} 
            </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">C=1000 & Gamma = 0.1</p>', unsafe_allow_html=True)    
   
    elif selected == "Manually-guide":
        st.markdown(""" <style> .font {
            font-size:20px ; font-family: 'Cooper Black'; color: #FF9633;} 
            </style> """, unsafe_allow_html=True)
            
        st.markdown('<p class="font">Focus on your favourite currency</p>', unsafe_allow_html=True)

        st.subheader('If you know SVM algorithm')
        st.markdown('If you want to focus on any currency, by selecting Manually in the SVM section , you can change C and Gamma Manually.Just need to select Manually checkbox in the app.')
    
    elif selected == 'Project-Report' :
        with open(url="https://github.com/mrs1425/PredictCrypto/blob/main/Project-Report.pdf", "rb") as file:
            st.download_button(
                    label="Download Project-Report",
                    data=file,
                    file_name="Project-Report.pdf",
                    )
            
st.sidebar.header('Input Data')

# Input function

def data():
    numb = st.sidebar.text_input('How many days do you wanna predict? ',5)
    stock_symbol = st.sidebar.selectbox('Select the symbol',['BTC-USD','XRP-USD','BCH-USD','USDT-USD','XEM-USD','ADA-USD','XTZ-USD','XMR-USD','LINK-USD','XLM-USD','BNB-USD','TRX-USD','LTC-USD','ETH-USD'])
    mydate = st.sidebar.selectbox('Select the start Date : ',['2017-01-01','2018-01-01','2019-01-01','2020-01-01','2021-01-01','2022-01-01'] )
    
    return stock_symbol,numb,mydate
    
        
symbol , n , mydate = data()



# In[]: get last online information of symbols from Yahoo Finance

def get_data():
    df = yf.Ticker(symbol)
    df = df.history(priod = "id",start = mydate)
    df['Date'] = df.index
    df = df.set_index(pd.DatetimeIndex(df['Date'].values))
    return df

df = get_data()


# In[Chart]:
    
st.header(symbol + ' / Close Price ')
st.line_chart(df['Close'])

st.header(symbol + '/ Data')
st.write(df.describe())


# In[Algorithms] :

df = df[['Close']]
forecast = int(n)
df['prediction'] = df[['Close']].shift(-forecast)
x = np.array(df.drop(['prediction'],1))
x = x[:-forecast]
y = np.array(df['prediction'])
y = y[:-forecast]

# In[]:

crypto = get_data()
st.header('Current Price')
st.warning(crypto.tail(1).Close)


xtrain , xtest , ytrain , ytest= train_test_split(x,y,test_size=0.2)

# Automatic & Manual Button

radio_val = st.radio("You can choose C & gamma in SVM Method: ",('Automatic','Manually'))

if  radio_val == 'Manually':
    st.info("You can change C and Gamma in the left side section")
    Cmanual = st.sidebar.text_input('I think C must :',1000)
    Cmanual = int(Cmanual)
    gamma_manual = st.sidebar.text_input('gamma must : ',0.1)
    gamma_manual = float(gamma_manual)
    

elif radio_val == 'Automatic' :
    Cmanual = 1000
    gamma_manual = 0.1
    
# A kernel is a function used in SVM for helping to solve problems. They provide shortcuts to avoid complex calculations. The amazing thing about kernel is that we can go to higher dimensions and perform smooth calculations with the help of it. We can go up to an infinite number of dimensions using kernels.
mysvr = SVR(kernel='rbf',C=Cmanual,gamma=gamma_manual)

mysvr.fit(xtrain, ytrain)

#Returns the coefficient of determination R^2 of the prediction. The coefficient R^2 is defined as 
svmconf = mysvr.score(xtest,ytest)

st.header('SVM Accuracy is : ')     
svmconf = "{:.0%}".format(svmconf)   # to write 2 digits in percentage
st.success(svmconf)

# In[]: SVM Prediction

x_forecast= np.array(df.drop(['prediction'],1))[-forecast:]
svmpred = mysvr.predict(x_forecast)
st.header('SVM Prediction')
st.success(svmpred)
svmpred

# In[]: LR Accuracy

lr = LinearRegression()
lr.fit(xtrain,ytrain)
lrconf = lr.score(xtest,ytest)
st.header('LR Accuracy is : ')
lrconf = "{:.0%}".format(lrconf)   # to write 2 digits in percentage
st.success(lrconf)

# In[]: LR Prediction
    
lrpred = lr.predict(x_forecast)
st.header('LR Prediction')
st.success(lrpred)
lrpred

# In[]: trendline


data=crypto
data=data.tail(90)
data0 = data.copy()
data0['date_id'] = ((data0.index.date - data0.index.date.min())).astype('timedelta64[D]')
data0['date_id'] = data0['date_id'].dt.days + 1
data1 = data0.copy()

while len(data1)>3:

    reg = linregress(
                    x=data1['date_id'],
                    y=data1['High'],
                    )
    data1 = data1.loc[data1['High'] > reg[0] * data1['date_id'] + reg[1]]

reg = linregress(
                    x=data1['date_id'],
                    y=data1['High'],
                    )

data0['high_trend'] = reg[0] * data0['date_id'] + reg[1]

data1 = data0.copy()

while len(data1)>3:

    reg = linregress(
                    x=data1['date_id'],
                    y=data1['Low'],
                    )
    data1 = data1.loc[data1['Low'] < reg[0] * data1['date_id'] + reg[1]]

reg = linregress(
                    x=data1['date_id'],
                    y=data1['Low'],
                    )

data0['low_trend'] = reg[0] * data0['date_id'] + reg[1]

# In[81]:

# The figsize attribute allows us to specify the width and height of a figure in-unit inches
plt.figure(figsize=(16,8))

data0['Close'].plot()
data0['high_trend'].plot()
data0['low_trend'].plot()
# plt.savefig('trendline.png')
plt.show()


# In[92]:


st.header('TRENDLINE : ')
st.pyplot()


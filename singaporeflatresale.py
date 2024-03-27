import streamlit as st  ## Streamlit library to build GUI interface
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np ## Numpy library to convert the values to array/log_transformation/transfer to exponential value
import pickle   ## This library is used to call the model which was build and tored
from sklearn import preprocessing  ##Machine Learning library for encoding
from sklearn.preprocessing import StandardScaler ##Machine Learning library for Scaling
le = preprocessing.LabelEncoder() ## Calling the encoding library to "le"

## Streamlit page config section
def page_config():
    st.set_page_config(layout= "wide")
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("https://rare-gallery.com/uploads/posts/568667-architecture.jpg");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.write("""
    <div style='text-align:center'>
        <h1 style='color:#20EEF7;'>Singapore Flat Resale Prediction</h1>
    </div>
    """,unsafe_allow_html=True)
    
page_config()

# https://media.istockphoto.com/id/1406528811/photo/dark-brown-rough-texture-toned-concrete-wall-surface-close-up-brown-background-with-space-for.jpg?s=612x612&w=0&k=20&c=KeT1jdiXSsrJjqk5-wlW_8DB-8nqWe4rU9JKZbmyF-4=

## Below code will create the header option menu  
selected = option_menu(None, ["Home","Flat_Resale_Prediction","Statistical Analysis"], 
                icons=["house","bar-chart-line","bar-chart-line"],
                menu_icon= "menu-button-wide",
                orientation= "horizontal",
                default_index=0,
                styles={"nav-link": {"font-size": "20px", "text-align": "centre", "margin": "-1px", "--hover-color": "#A7A405"},
                        "nav-link-selected": {"background-color": "#A7A405"}})

## Below will display the home tab section of the page
if selected == "Home":
    st.write(" ")
    st.markdown("### <span style='color:#20EEF7;'>Overview :</span>",
             unsafe_allow_html=True)
    st.markdown("#### <span style='color:#FFFFFD;'>This streamlit app aims to give users a friendly environment which can be used to predict Singapore Flat Resale Price in the future based on its past & present data. Since the resale flat market in Singapore is highly competitive, and it can be challenging to accurately estimate the resale value of a flat with many factors that are affecting the resale prices, such as location, flat type, floor area, and lease duration. In order to make it easy this a predictive model can help to overcome these challenges by providing users with an estimated resale price based on these factors.</span>",
             unsafe_allow_html=True)
    st.write(" ")
    st.markdown("### <span style='color:#20EEF7;'>Objective :</span>",
            unsafe_allow_html=True)
    st.markdown("### <span style='color:#FFFFFD;'>This project focuses on building Machine Learning algorithms to predict Flat_Resale_Price in Singapore Housing department using various libraries such as pandas, numpy, scikit-learn. The objective of the project is to preprocess the data, handle missing values, detect outliers, and handle skewness to correct and improve the model accuracy</span>",
             unsafe_allow_html=True)
    st.write(" ")
    col1 ,col2,col3=st.columns([1.5,1.5,1.5])
    with col1:
            st.write("#### <span style='color:#20EEF7;'>Technologies used</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#FFFFFD;'>- PYTHON   (PANDAS, NUMPY)</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#FFFFFD;'>- SCIKIT-LEARN</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#FFFFFD;'>- DATA PREPROCESSING</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#FFFFFD;'>- EXPLORATORY DATA ANALYSIS</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#FFFFFD;'>- STREAMLIT</span>",unsafe_allow_html=True)

    with col2:
            st.write("#### <span style='color:#20EEF7;'>Machine Learning Models used for this project</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#FFFFFD;'> - Linear Regression : :green[78% Accuracy]</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#FFFFFD;'> - KNN regressor : :green[92% Accuracy]</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#FFFFFD;'> - Decision Tree  : :green[80% Accuracy]</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#FFFFFD;'> - Decision Tree with Hyper Parameter Tuning : :green[96% Accuracy]</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#FFFFFD;'> - RandomForest Regressor (Hyper Parameter Tuning & Feature importance selection)  : :green[97% Accuracy]</span>",unsafe_allow_html=True)

    with col3:
            st.write("#### <span style='color:#20EEF7;'>Machine Learning Model</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#FFFFFD;'> - Which Model was selected? :red[RandomForestRegressor]</span>",unsafe_allow_html=True)
            st.write("#### <span style='color:#FFFFFD;'> - Why RandomForestRegressor?</span>",unsafe_allow_html=True)
            st.write("- RandomForestRegressor is an ensemble learning method which splits the training data differently every time for each Decision Tree's and takes the highest frequency positive value(highest purity) out of all to create a robust and accurate Regression model. This will also eliminate the overfitting issue between models. Hence this is one of the best to use for ML modles.")

elif selected == "Flat_Resale_Prediction":
    st.write("### :green[Singapore Flat Resale Prediction]")

    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("https://www.mydomaine.com/thmb/3PBtLuz6jOjzvkC3FKUYmsRO-PU=/1500x0/filters:no_upscale():strip_icc()/ForbesMasters-Moody-Guestroom-1-374c9e4c30c64d47a708713451104c2b.jpg");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    flat_type_dict = {'1 Room':0, '3 Room':1, '4 Room':2, '5 Room':3, '2 Room':4, 'Executive':5,'Multi Generation':6}
    town_dict = {'Ang Mo Kio':0,'Bedok':1,'Bishan':2,'Bukit Batok':3,'Bukit Merah':4,'Central Area':5,'Choa Chu Kang':6,'Clementi':7,'Geylang':8,'Hougang':9,'Jurong East':10,'Jurong West':11,'Kallang/Whampoa':12,'Marine Parade':13,'Queenstown':14,'Sengkang':15,'Serangoon':16,'Tampines':17,'Toa Payoh':18,'Woodlands':19,'Yishun':20,'Bukit Timah':21,'Sembawang':22,'Bukit Panjang':23,'Lim Chu Kang':24,'Pasir Ris':25,'Punggol':26}
    flat_model_dict = {'Improved':0,'New Generation':1,'Model A':2,'Standard':3,'Model A-Maisonette':4,'Apartment':5,'Maisonette':6,'Terrace':7,'Simplified':8,'2-Room':9,'Improved-Maisonette':10,'Multi Generation':11,'Premium Apartment':12,'Adjoined Flat':13,'Premium Maisonette':14,'Model A2':15,'Dbss':16,'Type S1':17,'Premium Apartment Loft':18,'3Gen':19}
    region_dict = {'North-East Region':0,'Eastern Region':1,'Central/Southern Region':2,'Western Region':3,'Northern Region':4}

    col1,col2= st.columns([1,1],gap="large")
    with col1:
            region_values = ['Central/Southern Region', 'Eastern Region', 'Northern Region', 'North-East Region','Western Region']
            region = st.selectbox(label='Select the specific Region', options=region_values)

            if region == 'Central/Southern Region':
                town_values = ['Bishan', 'Bukit Merah', 'Central Area', 'Geylang',
                               'Kallang/Whampoa', 'Marine Parade', 'Queenstown', 'Toa Payoh','Bukit Timah']
                town = st.selectbox(label='Select the Town', options=town_values)
            elif region == 'Eastern Region':
                town_values = ['Bedok', 'Tampines', 'Pasir Ris']
                town = st.selectbox(label='Select the Town', options=town_values)
            elif region == 'Northern Region':
                town_values = ['Woodlands', 'Yishun', 'Sembawang', 'Lim Chu Kang']
                town = st.selectbox(label='Select the Town', options=town_values)
            elif region == 'North-East Region':
                town_values = ['Ang Mo Kio', 'Hougang', 'Sengkang', 'Serangoon', 'Punggol']
                town = st.selectbox(label='Select the Town', options=town_values)
            elif region == 'Western Region':
                    town_values = ['Bukit Batok', 'Choa Chu Kang', 'Clementi', 'Jurong East',
                                    'Jurong West', 'Bukit Panjang']
                    town = st.selectbox(label='Select the Town', options=town_values)
            flat_type_values = ['1 Room', '3 Room', '4 Room', '5 Room', '2 Room', 'Executive',
                                'Multi Generation']
            Flat = st.selectbox(label='Select the Flat Type', options=flat_type_values)
            remaining_years = st.number_input(label="Enter the lease remaining years",min_value=1.00, max_value=99.0, value=1.0)

    with col2:
            flat_model_values = ['Improved', 'New Generation', 'Model A', 'Standard',
                                 'Model A-Maisonette', 'Apartment', 'Maisonette', 'Terrace',
                                 'Simplified', '2-Room', 'Improved-Maisonette', 'Multi Generation',
                                 'Premium Apartment', 'Adjoined Flat', 'Premium Maisonette',
                                 'Model A2', 'Dbss', 'Type S1', 'Premium Apartment Loft', '3Gen']
            flat_model = st.selectbox(label='Select the Flat Model', options=flat_model_values)
            floor_area_sqm = st.number_input(label="Enter the sqm area",min_value=1.00, max_value=100000.0, value=1.0)
            Month_values = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12]
            Month = st.selectbox(label='Month', options=Month_values)
            year_values = [1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
                           2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011,
                           2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022,
                           2023, 2024]
            Year = st.selectbox(label='Registered Year', options=year_values)
            Floor = st.number_input(label='Floor level ', min_value=1.0, max_value=100.0, value=1.0)
    
    
    if st.button("Predict Price"):
        with open('scaled.pkl', 'rb') as f:
            regg1_SS = pickle.load(f)

        # try:
        x = regg1_SS.transform([[town_dict[town],flat_type_dict[Flat],floor_area_sqm,flat_model_dict[flat_model],remaining_years,Month,Year,Floor,region_dict[region]]])
        ## Below code will call the buid model and assign to Class1 Varaible 
        
        with open('HBD.pkl', 'rb') as f:
            regg1 = pickle.load(f)

        pred = regg1.predict(x)   ## Predicting the value with assigned model variable regg
        st.markdown(f"### :orange[Predicted resale Price is] :green[{round(pred[0],2)}]")
        st.info("Note: Price is predicted based on past data. In real time it can defer based on circumstances")
        # except:
        #         st.error("Please enter only numeric values to 'Country' & 'Quantity Tons' ")

elif selected == "Statistical Analysis":
     st.write("### :green[Welcome to Statistical Analysis on Singapore Housing Department]")


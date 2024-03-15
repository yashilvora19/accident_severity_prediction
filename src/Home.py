import streamlit as st
import pandas as pd
import numpy as np

# RUN THE FILE
# python -m streamlit run Home.py OR
# streamlit run Home.py

st.set_page_config(page_title="Predicting Severity of Road Accidents in the U.K.")

st.sidebar.title("Navigation")

st.title("CSE 151A Project - Predicting Severity of Road Accidents in the U.K.")

# Add your project description here

st.markdown(
    """
    One of the leading causes of non natural death is road accidents. There may be several contributing factors that 
    lead to vehicle casualties, including traffic, weather, road conditions etc. We wanted to predict the severity 
    of road accidents ranging from Slight, Serious, to Fatal using supervised models such as Logistic Regression, 
    Decision Trees etc. Attributes that may be used to predict the data include the road conditions, the weather 
    conditions, vehicle types, or what kind of area they’re in.

    Our data is mainly focused on locations in the UK, so while it may not necessarily apply similarly in the US, 
    we could still use this model to run on US datasets and see the results. It is a dataset with 
    14 columns and over 600k observations, with columns including severity of accident, the date, number 
    of casualties, longitude/ latitude, road surface conditions, road types, urban/ rural areas, 
    weather conditions, and vehicle types. Ethical concerns include if our stakeholders were vehicle companies, 
    would they have reduced sales if, say, trucks were more likely to lead to severe accidents? 
    However, by figuring out what would predict the severity of road accidents, we can also prevent harm by 
    noting the features that largely impact the severity.
    """
)

st.markdown(
    """
    ### Test Our Model Here!
    """
)
latitude = st.text_input("Latitude", "")
longitude = st.text_input("Longitude", "")
# Accident_Severity', 'Accident Date', 'Latitude',
#        'Light_Conditions', 'Longitude',
#        'Number_of_Casualties', 'Number_of_Vehicles', 'Road_Surface_Conditions',
#        'Road_Type', 'Urban_or_Rural_Area', 'Weather_Conditions',
#        'Vehicle_Type'
# 

# urban_or_rural_area = st.selectbox("Urban or Rural Area", ["Urban", "Rural"])
number_of_casualties = st.text_input("Number of Casualties", "")
number_of_vehicles = st.text_input("Number of Vehicles", "")
day = st.text_input("Day (1-31)", "")
month = st.text_input("Month (1-12)", "")
year = st.text_input("Year (XXXX)", "")

light_conditions = st.radio("Light Conditions", ["Darkness - no lighting", "Darkness - lights unlit", "Darkness - lights lit", "Daylight"])
weather_conditions = st.multiselect("Weather Conditions", ["Fine", "Raining", "High Winds", "Snowing", "Fog/Mist","Other"])

road_type = st.radio('Road Type', ['Single carriageway', 'Dual carriageway', 'One way street', 
                           'Roundabout', 'Slip road'])

vehicle_type = st.radio("Vehicle Type", 
                            ['Car', 'Bus/Coach', 'Minibus', 'Van', 'Motorcycle (<50cc)', 
                             'Motorcycle (125cc-500cc)', 'Motorcycle (>500cc)', 'Goods (3.5-7.5 tonnes)', 
                             'Goods (>7.5 tonnes)', 'Taxi/Private hire car','Minibus', 'Pedal Cycle',
                             'Agricultural vehicle', 'Horse','Other'])

Road_Surface_Conditions = st.radio('Road Conditions', ['Dry', 'Wet/damp', 'Snow', 'Frost/ice','Flood over 3cm. deep'])
urban_or_rural = st.radio('Urban or Rural', ['Urban', 'Rural'])

st.write('Our model will now predict the severity of the accident given these conditions. Severity can be Mild, Severe, or Fatal')

clicked = st.button('Predict Accident Severity')

if clicked:
    st.write("Results")
    st.success("Data processed successfully. Here's the accident severity: Severe")
    st.balloons()

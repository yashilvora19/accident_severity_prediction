import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

# RUN THE FILE
# python -m streamlit run Home.py OR
# streamlit run Home.py

st.set_page_config(page_title="Predicting Severity of Road Accidents in the U.K.")

path = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, 'models', 'saving_model.keras')
    model = tf.keras.models.load_model(model_path)
    return model
model = load_model()

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

# urban_or_rural_area = st.selectbox("Urban or Rural Area", ["Urban", "Rural"])
number_of_casualties = st.text_input("Number of Casualties", "")
number_of_vehicles = st.text_input("Number of Vehicles", "")
day = st.text_input("Day (1-31)", "")
month = st.text_input("Month (1-12)", "")
year = st.text_input("Year (XXXX)", "")

light_conditions = st.radio("Light Conditions", ["Darkness - no lighting", "Darkness - lights unlit", "Darkness - lights lit", "Daylight"])

weather_conditions = st.multiselect("Weather Conditions", ['Fine','Fog/mist', 'High winds', 'Other', 'Raining', 'Snowing'])

road_type = st.radio('Road Type', ['Dual carriageway', 'One way street', 'Roundabout','Single carriageway', 'Slip road'])

vehicle_type = st.radio("Vehicle Type", 
                        ['Agricultural Vehicle', 'Bus/coach', 'Car',
                        'Goods (>7.5 tonnes mgw', 'Goods (3.5t-7.5t', 'Minibus', 'Motorcycle (<125cc)',
                        'Motorcycle (<50cc and under)', 'Motorcycle (125cc-500cc)',
                        'Motorcycle (>500cc)', 'Other vehicle', 'Pedal cycle', 'Ridden horse',
                        'Taxi/Private hire car', 'Van / Goods 3.5 tonnes mgw or under'])
                            

Road_Surface_Conditions = st.radio('Road Conditions', ['Dry', 'Flood over 3cm. deep', 'Frost/ice', 'Snow', 'Wet/damp'])

urban_or_rural = st.radio('Urban or Rural', ['Urban', 'Rural'])

st.markdown(
    """
    **Our model will now predict the severity of the accident given these conditions.** 
    Severity can be:
     - Mild, 
     - Severe 
     - Fatal.
    """
    )

clicked = st.button('Predict Accident Severity')
# """
# Index(['Latitude', 
#     'Light_Conditions'- LABEL ENCODED 
# 'Longitude',
# 'Number_of_Casualties'
# 'Number_of_Vehicles'
# 'Day'
# 'Month'
# 'Year'

# ROAD CONDITIONS- 'Dry', 'Flood over 3cm. deep', 'Frost or ice', 'Snow', 'Wet or damp',
# ROAD TYPE -'Dual carriageway', 'One way street', 'Roundabout',
#        'Single carriageway', 'Slip road', 
# RURAL UNALLOCATED URBAN

# VEHICLE TYPE-----'Agricultural vehicle', 'Bus or coach (17 or more pass seats)', 'Car',
#        'Goods 7.5 tonnes mgw and over', 'Goods over 3.5t. and under 7.5t',
#        'Minibus (8 - 16 passenger seats)', 'Motorcycle 125cc and under',
#        'Motorcycle 50cc and under', 'Motorcycle over 125cc and up to 500cc',
#        'Motorcycle over 500cc', 'Other vehicle', 'Pedal cycle', 'Ridden horse',
#        'Taxi/Private hire car', 'Van / Goods 3.5 tonnes mgw or under', 
# WAETHER CONDITIONS----'Fine','Fog or mist', 'High winds', 'Other', 'Raining', 'Snowing',
# """

if clicked:
    # LIGHT CONDITIONS
    conditions = ["Darkness - no lighting", "Darkness - lights unlit", "Darkness - lights lit", "Daylight"]
    light_conditions = conditions.index(light_conditions)
    
    # WEATHER CONDITIONS
    weather_conditions_list = ['Fine','Fog/mist', 'High winds', 'Other', 'Raining', 'Snowing']
    weather_conditions_arr = np.zeros(6)
    for i in range(len(weather_conditions)):
        for j in range(len(weather_conditions_list)):
            if (weather_conditions[i] == weather_conditions_list[j]):
                weather_conditions_arr[j] == 1

    road_type_list = ['Dual carriageway', 'One way street', 'Roundabout','Single carriageway', 'Slip road']
    road_types_arr =np.zeros(5)
    road_types_arr[road_type_list.index(road_type)] = 1

    vehicle_type_list = ['Agricultural Vehicle', 'Bus/coach', 'Car',
                            'Goods (>7.5 tonnes mgw', 'Goods (3.5t-7.5t', 'Minibus', 'Motorcycle (<125cc)',
                            'Motorcycle (<50cc and under)', 'Motorcycle (125cc-500cc)',
                            'Motorcycle (>500cc)', 'Other vehicle', 'Pedal cycle', 'Ridden horse',
                            'Taxi/Private hire car', 'Van / Goods 3.5 tonnes mgw or under']
    vehicle_types_arr =np.zeros(len(vehicle_type_list))
    vehicle_types_arr[vehicle_type_list.index(vehicle_type)] = 1


    road_conds_list = ['Dry', 'Flood over 3cm. deep', 'Frost/ice', 'Snow', 'Wet/damp']
    road_conditions =np.zeros(5)
    road_conditions[road_conds_list.index(Road_Surface_Conditions)] = 1

    # URBAN AND RURAL ENCODING
    urban_rural_arr = np.zeros(3)

    if urban_or_rural=='Rural':
        urban_rural_arr[0] == 1
    else:
        urban_rural_arr[2] == 1
    
    # encoded and concatenating arrays
    input_encoded = np.array([latitude, light_conditions, longitude, number_of_casualties, number_of_vehicles, day, month, year])
    input_encoded = np.concatenate((input_encoded, road_conditions))
    input_encoded = np.concatenate((input_encoded, road_types_arr))
    input_encoded = np.concatenate((input_encoded, urban_rural_arr))
    input_encoded = np.concatenate((input_encoded, vehicle_types_arr))
    input_encoded = np.concatenate((input_encoded, weather_conditions_arr))
    
    # road_conditions, road_types_arr, urban_rural_arr, vehicle_types_arr, weather_conditions_arr

    final_input = input_encoded.reshape(1,-1)

    input = np.zeros((1,42))
    out = model.predict(final_input.astype(float))

    if np.argmax(out) == 0:
        severity = 'Mild'
    elif np.argmax(out) == 1:
        severity = 'Severe'
    else:
        severity = 'Fatal'

    st.write("Results")
    st.success("Data processed successfully. Here's the accident severity: " + severity)
    st.balloons()

    del input
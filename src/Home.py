import streamlit as st
import pandas as pd
import numpy as np

# RUN THE FILE
# python -m streamlit run app.py OR
# streamlit run app.py
st.set_page_config(
    page_title="Welcome to our app!"
)

st.sidebar.header("Our Model")

st.sidebar.title("Options")
option = st.sidebar.selectbox("Select an option:", ["About", "Predict"])

if option == "About":
    st.header("About")
    st.write("This app predicts the severity of road accidents based on various factors such as road conditions, weather conditions, vehicle types, etc.")

    st.subheader("Ethical Concerns")
    st.write("There may be ethical concerns if stakeholders such as vehicle companies are involved. By predicting accident severity, we can work towards preventing harm by understanding the factors that largely impact severity.")

    st.subheader("Data Source")
    st.write("The dataset used in this app mainly focuses on locations in the UK, but the model can be applied to datasets from other regions as well.")


st.title('CSE 151A Project- Accident Severity Prediction')

st.sidebar.success("Want to learn more about our project?")

# # path = os.path.dirname(__file__)
# accident = '../imgs/accident.jpg'
# image1 = Image.open(accident)


st.markdown(
    """
    One of the leading causes of non natural death is road accidents. There may be several contributing factors that 
    lead to vehicle casualties, including traffic, weather, road conditions etc. We wanted to predict the severity 
    of road accidents ranging from Slight, Serious, to Fatal using supervised models such as Logistic Regression, 
    Decision Trees etc. Attributes that may be used to predict the data include the road conditions, the weather 
    conditions, vehicle types, or what kind of area they’re in.

    Our data is mainly focused on locations in the UK, so while it may not necessarily apply similarly in the US, we could still use this model to run on US datasets and see the results. It is a dataset with 14 columns and over 600k observations, with columns including severity of accident, the date, number of casualties, longitude/ latitude, road surface conditions, road types, urban/ rural areas, weather conditions, and vehicle types. Ethical concerns include if our stakeholders were vehicle companies, would they have reduced sales if, say, trucks were more likely to lead to severe accidents? However, by figuring out what would predict the severity of road accidents, we can also prevent harm by noting the features that largely impact the severity.
    """
)

# st.image(image1)

st.markdown(
    """
    ### Test Our Model
    """
)

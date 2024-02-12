# Predicting Severity of Road Accidents

### Abstract 
One of the leading causes of non natural death is road accidents. There may be several contributing factors that lead to vehicle casualties, including traffic, weather, road conditions etc. We wanted to predict the severity of road accidents ranging from Slight, Serious, to Fatal using supervised models such as Logistic Regression, Decision Trees etc. Attributes that may be used to predict the data include the road conditions, the weather conditions, vehicle types, or what kind of area they’re in. 

Our data is mainly focused on locations in the UK, so while it may not necessarily apply similarly in the US, we could still use this model to run on US datasets and see the results. It is a dataset with 14 columns and over 600k observations, with columns including severity of accident, the date, number of casualties, longitude/ latitude, road surface conditions, road types, urban/ rural areas, weather conditions, and vehicle types. Ethical concerns include if our stakeholders were vehicle companies, would they have reduced sales if, say, trucks were more likely to lead to severe accidents? However, by figuring out what would predict the severity of road accidents, we can also prevent harm by noting the features that largely impact the severity. 

### Visualization Steps

We made use of the pairplot and heatmap with correlation matrix in order to get an overall sense of the distribution of data. Through this we were also able to get correlations between different features and became one with the data. We were able to see trends and patterns through a few more visualizations of light intensities VS number of accidents, number of casualties VS number of vehicles, and how the number of accidents varied over the past 4 years. 

All of this collectively gave us a better idea of what the data looks like which in turn gives us a better idea of which models to use in the next step of this project.

### Preprocessing Steps
We already began preprocessing the data by one hot encoding most of the categorical data (`Road_Surface_Conditions`, `Road_Type`, `Urban_or_Rural_Area`, `Vehicle_Type`), and for `Light_Conditions`, we chose to make it ordinal and encode it from 0 for Dark, and 3 for Daylight. We selected MultiBinarizer to do multiple one-hot encoding for each row for the `Weather Conditions` as there were multiple categories that were satistified. Then we chose to normalize the `Latitude` and `Longitude` to make it a more contained value. We left the `Number_of_Casualties` and `Number_of_Vehicles` as is, as the values were just integers and seemed to have no large outliers.

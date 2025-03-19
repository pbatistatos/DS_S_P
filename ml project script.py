# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 23:14:51 2025

@author: Administrator
"""

import pandas as pd
import duckdb as db
import matplotlib as plt 
import seaborn as sns
import random
import time
import missingno as msno

from geopy.distance import geodesic
import numpy as np
from scipy.stats import f_oneway
import statsmodels.api as sm 
from statsmodels.formula.api import ols 

from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#for twilight variables imputation
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
import pytz
from suntime import Sun, SunTimeException

#to impute missing zipcodes
from geopy.geocoders import Nominatim

# Haversine formula to compute destination coordinates (bearing = direction of travel)
def compute_end_coordinates(start_latit, start_long, distance, bearing):
   
    R = 6371  # Radius of the Earth in kilometers
    lat1 = np.radians(start_latit)
    lon1 = np.radians(start_long)
    angular_distance = distance / R  # Distance in radians

    # Calculate the end latitude and longitude
    lat2 = np.arcsin(np.sin(lat1) * np.cos(angular_distance) + 
                     np.cos(lat1) * np.sin(angular_distance) * np.cos(bearing))

    lon2 = lon1 + np.arctan2(np.sin(bearing) * np.sin(angular_distance) * np.cos(lat1),
                             np.cos(angular_distance) - np.sin(lat1) * np.sin(lat2))

    # Convert back to degrees
    lat2 = np.degrees(lat2)
    lon2 = np.degrees(lon2)

    return lat2, lon2

def complete_latitude(row):
    if pd.isna(row['End_Lat']) == 1 and row['Distance(mi)']<=2:
        return row['Start_Lat']
    elif pd.isna(row['End_Lat']) == 1 and row['Distance(mi)']>=2:
        bearing = np.radians(270)
        end_lat, end_lon = compute_end_coordinates(row['Start_Lat'], row['Start_Lng'], row['Distance(mi)'], bearing) 
        return end_lat
    else:
        return row['End_Lat']

def complete_logtitude(row):
    if pd.isna(row['End_Lng']) == 1 and row['Distance(mi)']<=2:
        return row['Start_Lng']
    elif pd.isna(row['End_Lng']) == 1 and row['Distance(mi)']>=2:
        bearing = np.radians(270)
        end_lat, end_lon = compute_end_coordinates(row['Start_Lat'], row['Start_Lng'], row['Distance(mi)'], bearing) 
        return end_lon
    else:
        return row['End_Lng']
    
def keep_the_right_WC(row):
    if pd.isna(row['Wind_Chill(F)']) and row['Temperature(F)_n']> 50:
        return row['Temperature(F)_n']
    elif pd.isna(row['Wind_Chill(F)']) and row['Temperature(F)_n']<= 50:
        return row['Wind_Chill(F)_n']
    else:
        return row['Wind_Chill(F)_n']

def fix_and_convert_datetime(row):    
    return str(row['Start_Time'])[:19]
    
def get_date(row):
    return str(row['Start_Time'])[:10]

def get_time(row):
    return str(row['Start_Time'])[11:19]

def count_minutes_past_midnight(row):
    return (int(str(row['time_n'])[:2])*60) + int(str(row['time_n'])[3:5])
'''
def count_days_past_eve(row):
    return (int(str(row['date_n'])[6:8])*60) + int(str(row['date_n'])[3:5])
    '''
def day_of_the_year(row):
    year = str(row['date_n'])[:4]
    month = str(row['date_n'])[5:7]
    day = str(row['date_n'])[8:10]
    return datetime(int(year),int(month),int(day)).timetuple().tm_yday   
    
    
start_df = pd.read_csv(r"C:\Users\Administrator\Desktop\Έγγραφα Μτεαπτυχιακού 2\Σημειώσεις Μεταπρυχιακού\ml\US_Accidents_March23.csv")

'''
lista =[]
while len(lista) <100000:
    n = random.randint(0,len(start_df))
    if n not in lista:
        lista.append(n)
        
        
df_to_work = pd.DataFrame(columns = start_df.columns)        
for i in lista:
    df_to_work = pd.concat([df_to_work,start_df.iloc[[i]]], ignore_index = True)
'''
df_to_work = start_df.sample(n=100000)
df_to_work.to_excel(r"C:\Users\Administrator\Desktop\Έγγραφα Μτεαπτυχιακού 2\Σημειώσεις Μεταπρυχιακού\ml\New Microsoft Excel Worksheet.xlsx")

df_to_work = pd.read_excel(r"C:\Users\Administrator\Desktop\Έγγραφα Μτεαπτυχιακού 2\Σημειώσεις Μεταπρυχιακού\ml\New Microsoft Excel Worksheet.xlsx")

descriptives_df = df_to_work.describe()
infor = df_to_work.info()

msno.heatmap(df_to_work)
msno.bar(df_to_work)

missing_columns = []
for i in df_to_work.columns:
    if df_to_work[i].isnull().sum()!=0:
        missing_columns.append(i)
missing_df = df_to_work[missing_columns]
msno.bar(missing_df)



'''
for i in df_to_work.columns:
    sns.boxplot(data = df_to_work[df_to_work.columns[5]],log_scale =10)
    #time.sleep(5)
    #go = input(str("please write 'go'"))
'''

df_to_work.columns

df_to_work = df_to_work.assign(comp_latitude_1=df_to_work.apply(complete_latitude, axis=1))

df_to_work = df_to_work.assign(comp_longtitude_1=df_to_work.apply(complete_logtitude, axis=1))

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply the Z-score standardization (fit_transform standardizes the data)
standardized_data = scaler.fit_transform(df_to_work[['Distance(mi)','Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']])

# Convert the standardized data back to a DataFrame
standardized_df = pd.DataFrame(standardized_data, columns=df_to_work[['Distance(mi)','Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)']].columns)

matrix1 = standardized_df.corr()

matrix = df_to_work[['Start_Lat', 'Start_Lng', 'Distance(mi)','Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)','comp_latitude_1', 'comp_longtitude_1']].corr()


#sm.stats.anova_lm(model, type=2)
'''
# to calculate missing values for geospacial data - NOT WORKING because of non float variables 
cols = ['Start_Lat', 'Start_Lng','Street', 'City', 'County', 'State', 'Zipcode', 'Country', 'Timezone', 'Airport_Code']
X = df_to_work[cols]
X.info()

impute_it = IterativeImputer()
impute_it.fit_transform(X)
'''

cols = ['Start_Lat', 'Start_Lng', 'Distance(mi)','Temperature(F)', 'Wind_Chill(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)','comp_latitude_1', 'comp_longtitude_1']
X = df_to_work[cols]
X.info()

impute_it = IterativeImputer()
comp_cont_df = pd.DataFrame(impute_it.fit_transform(X))
comp_cont_df.info()

comp_cont_df = comp_cont_df.set_axis(['Start_Lat_n', 'Start_Lng_n', 'Distance(mi)_n','Temperature(F)_n', 'Wind_Chill(F)_n', 'Humidity(%)_n', 'Pressure(in)_n', 'Visibility(mi)_n', 'Wind_Speed(mph)_n', 'Precipitation(in)_n','comp_latitude_1_n', 'comp_longtitude_1_n'], axis="columns")

comp_cont_df.to_excel(r"C:\Users\Administrator\Desktop\Έγγραφα Μτεαπτυχιακού 2\Σημειώσεις Μεταπρυχιακού\ml\comp_cont_df.xlsx")


joined_df1 = df_to_work.join(comp_cont_df)
joined_df1.columns

joined_df1 = joined_df1.assign(final_wind_chill=joined_df1.apply(keep_the_right_WC, axis=1))
joined_df1 = joined_df1[['Unnamed: 0', 'ID', 'Source', 'Severity', 'Start_Time', 'End_Time',
       'Start_Lat', 'Start_Lng', 'Description', 'City', 'County', 'State', 'Zipcode',
       'Country', 'Timezone', 'Weather_Timestamp',
       'Wind_Direction', 'Weather_Condition', 'Amenity', 'Bump', 'Crossing',
       'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
       'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',
       'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight',
       'Astronomical_Twilight', 'comp_latitude_1', 'comp_longtitude_1',
       'Distance(mi)_n', 'Temperature(F)_n',
       'Humidity(%)_n', 'Pressure(in)_n',
       'Visibility(mi)_n', 'Wind_Speed(mph)_n', 'Precipitation(in)_n',
       'final_wind_chill']]
joined_df1.info()
msno.heatmap(joined_df1)

#work on missing values twilights
'''we will implement a classifier that classifies each missing record to day or night
using the variables start time, timezone, 
'''

#transformation of twilight variables to 1-0 
joined_df1['Sunrise_Sunset'] = joined_df1['Sunrise_Sunset'].str.replace("Day","1")
joined_df1['Sunrise_Sunset'] = joined_df1['Sunrise_Sunset'].str.replace("Night","0")

joined_df1['Civil_Twilight'] = joined_df1['Civil_Twilight'].str.replace("Day","1")
joined_df1['Civil_Twilight'] = joined_df1['Civil_Twilight'].str.replace("Night","0")

joined_df1['Nautical_Twilight'] = joined_df1['Nautical_Twilight'].str.replace("Day","1")
joined_df1['Nautical_Twilight'] = joined_df1['Nautical_Twilight'].str.replace("Night","0")

joined_df1['Astronomical_Twilight'] = joined_df1['Astronomical_Twilight'].str.replace("Day","1")
joined_df1['Astronomical_Twilight'] = joined_df1['Astronomical_Twilight'].str.replace("Night","0")

#transform time variables to datetime



joined_df1['Start_Time']= joined_df1['Start_Time'].astype(str)
joined_df1 = joined_df1.assign(datetime_n=joined_df1.apply(fix_and_convert_datetime, axis=1))

joined_df1 = joined_df1.assign(date_n=joined_df1.apply(get_date, axis=1))

joined_df1 = joined_df1.assign(time_n=joined_df1.apply(get_time, axis=1))

joined_df1 = joined_df1.assign(mins_past_midn=joined_df1.apply(count_minutes_past_midnight, axis=1))

joined_df1 = joined_df1.assign(day_of_the_year=joined_df1.apply(day_of_the_year, axis=1))


#joined_df1['datetime_n'] = pd.to_datetime(joined_df1['datetime_n'],errors='coerce')



#joined_df1 = joined_df1.assign(final_wind_chill=joined_df1.apply(keep_the_right_WC, axis=1))
# Apply the function to calculate day/night for known rows
#joined_df1['predicted_Sunrise_Sunset'] = joined_df1.apply(lambda row: calculate_day_night(row) if pd.isna(row['Sunrise_Sunset']) else row['Sunrise_Sunset'], axis=1)


joined_df1.columns
# Separate rows with known and unknown day/night values
known_data = joined_df1.dropna(subset=['Civil_Twilight'])
unknown_data = joined_df1[joined_df1['Civil_Twilight'].isna()]

# Train a model using known data
X = known_data[['Start_Lng', 'Start_Lat', 'Timezone', 'mins_past_midn', 'day_of_the_year' ]]
y = known_data['Civil_Twilight']

# Convert categorical data (time_zone) to numerical (One-Hot Encoding)
X = pd.get_dummies(X, columns=['Timezone'])

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
X.dtypes
# Prepare the unknown data
X_unknown = unknown_data[['Start_Lng', 'Start_Lat', 'Timezone', 'mins_past_midn', 'day_of_the_year' ]]
X_unknown = pd.get_dummies(X_unknown, columns=['Timezone'])

# Make predictions for the missing day/night values
predicted_values = model.predict(X_unknown)

# Fill the missing values
joined_df1.loc[joined_df1['Civil_Twilight'].isna(), 'Civil_Twilight'] = predicted_values



#NOT EXECUTABLE the following was used before introducing the calculation of longitutde and latitude of the 1032 missing values
#
#'''
#df_to_work['comp_longtitude_1'].dtypes()
#db.sql("CREATE TEMP TABLE t1 AS SELECT * FROM df_to_work A WHERE A.comp_longtitude_1 = 'Plese_check_it' ")
#df_check_1 = db.sql("SELECT * FROM t1").df()
#db.sql("DROP TABLE t1")

#df_check_1 = df_check_1.reset_index()
#df_check_1.to_excel(r"C:\Users\Administrator\Desktop\Έγγραφα Μτεαπτυχιακού 2\Σημειώσεις Μεταπρυχιακού\ml\missing values long_lat 1032.xlsx")
#'''

# Initialize geolocator
geolocator = Nominatim(user_agent="us_accidents_imputation")
list_of_errors=[]
def get_zipcode_from_coords(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), language='en')
        if location and 'address' in location.raw:
            address = location.raw['address']
            return address.get('postcode', None)
        return None
    except Exception as e:
        list_of_errors.append(e)
        print(f"Error for coordinates {lat}, {lon}: {e}")
        return None



# Apply the reverse geocoding function to your dataframe
joined_df1['Zipcode'] = joined_df1.apply(lambda row: get_zipcode_from_coords(row['Start_Lat'], row['Start_Lng']) if pd.isna(row['Zipcode']) else row['Zipcode'], axis=1)


mis_zip = joined_df1[pd.isna(joined_df1['Zipcode'])]
mis_zip['Zipcode'] = mis_zip.apply(lambda row: get_zipcode_from_coords(row['Start_Lat'], row['Start_Lng']) if pd.isna(row['Zipcode']) else row['Zipcode'], axis=1)


joined_df1.to_excel(r"C:\Users\Administrator\Desktop\Έγγραφα Μτεαπτυχιακού 2\Σημειώσεις Μεταπρυχιακού\ml\incomplete_dataset.xlsx")

df_to_start = pd.read_excel(r"C:\Users\Administrator\Desktop\Έγγραφα Μτεαπτυχιακού 2\Σημειώσεις Μεταπρυχιακού\ml\incomplete_dataset.xlsx")

#για να αποφευχθεί ο θόρυβος θα γίνει διαγραφή των εγγραφών που δεν καταφέραμε να υπολογίζουμε το zipcode από τις συντεταγμένες
df_to_start = df_to_start.dropna(subset=['Zipcode'])

df_to_start.info()
df_to_start.columns
mis_City = df_to_start[pd.isna(df_to_start['City'])]

#df_to_start['City'] = df_to_start.groupby('Zipcode')['City'].transform(lambda x: x.fillna(x.mode()))
#
#
#
#city,weather_timestamp,timezone  πέταμα

df_to_start['Timezone'] = df_to_start.groupby('County')['Zipcode'].transform(lambda x: x.fillna(x.mode()[0]))

df_to_start['Wind_Direction'] = df_to_start['Wind_Direction'].fillna('Unknown')
df_to_start['Weather_Condition'] = df_to_start['Weather_Condition'].fillna('Unknown')


df_to_start = df_to_start[['ID', 'Source', 'Severity', 'Start_Time',
       'End_Time', 'Start_Lat', 'Start_Lng', 'Description', 'County',
       'State', 'Zipcode', 'Country', 'Timezone', 'Wind_Direction', 'Weather_Condition', 'Amenity', 'Bump', 'Crossing',
       'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station',
       'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop',
       'Civil_Twilight', 'comp_latitude_1', 'comp_longtitude_1',
       'Distance(mi)_n', 'Temperature(F)_n', 'Humidity(%)_n', 'Pressure(in)_n',
       'Visibility(mi)_n', 'Wind_Speed(mph)_n', 'Precipitation(in)_n',
       'final_wind_chill', 'datetime_n', 'date_n', 'time_n', 'mins_past_midn',
       'day_of_the_year']]

df_to_start.to_excel(r"C:\Users\Administrator\Desktop\Έγγραφα Μτεαπτυχιακού 2\Σημειώσεις Μεταπρυχιακού\ml\no_missing_dataset.xlsx")













































#!/usr/bin/env python
# coding: utf-8

# # <center>Coursera Capstone Project for <br>IBM Data Science with Python</center>

# #### <center>Lutz Wimmer</center>

# ## Table of Contents:
# * [Assignment 1](#assignment1)
# * [Assignment 2](#assignment2)
#     * [Part 1](#part2.1)
#     * [Part 2](#part2.2)
#     * [Part 3](#part2.3)
# * [Assignment 3](#assignment3)
#     * [Part 1](#part3.1)
#     * [Part 2](#part3.2)

# ## <br><br>Assignment 1 - Capstone Project Notebook <a class="anchor" id="assignment1"></a>
# 

# <br>This Notebook is part of the Capstone Project for the "IBM Data Science with Python" course on Coursera.

# In[1]:


# import Python libraries
import pandas as pd
import numpy as np
print("Hello Capstone Project Course!")


# ## <br><br>Assignment 2 - Segmenting and Clustering Neighborhoods in Toronto <a class="anchor" id="assignment2"></a>

# ### <br>Part 1 - Web scraping list of postal codes of Toronto, Canada <a class="anchor" id="part2.1"></a>

# In[2]:


from bs4 import BeautifulSoup
import requests
import pandas as pd


# In[3]:


# saving webpage to memory as text
url = "https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M"
webtext = requests.get(url).text
webpage = BeautifulSoup(webtext,"html5lib")

webtab = webpage.find("table",class_='wikitable sortable')


# In[4]:


# creating pandas df
toronto = pd.DataFrame(columns=["PostalCode","Borough","Neighborhood"])

for row in webtab.tbody.find_all("tr"):
    col = row.find_all("td")
    if (col != []):
        pc = col[0].text.strip()
        bor = col[1].text.strip()
        nei = col[2].text.strip()
        toronto = toronto.append({"PostalCode": pc, "Borough": bor, "Neighborhood": nei}, ignore_index=True)

# filter unassigned values
fil = toronto.index[toronto['Borough'] == "Not assigned"].tolist()
toronto = toronto.drop(fil, axis = 0).reset_index(drop = True)

# checking for duplicate postal codes and "Not assigned" Neighborhoods
# toronto.iloc[:,0].duplicated().describe()
# toronto.iloc[:,2] == "Not assigned"
# none found
toronto


# In[5]:


toronto.shape


# ### <br>Part 2 - Assigning lat/long coordinates <a class="anchor" id="part2.2"></a>

# Using the link to the .csv file, the geocoder doesn't work.

# In[6]:


coords = pd.read_csv('https://cocl.us/Geospatial_data')
coords.shape


# In[7]:


# merging the two dataframes by PostalCode
coords.rename(columns={'Postal Code':'PostalCode'}, inplace = True)
toronto = toronto.merge(coords, on = 'PostalCode')
toronto


# ### <br>Part 3 - Clustering neighborhoods <a class="anchor" id="part2.3"></a>

# using ".*Toronto*." to filter out any boroughs outside the city.

# In[8]:


city = toronto[toronto['Borough'].str.match(".*Toronto*.")]
city['Borough'].unique()


# The number of neighborhoods in the city is 39 and thus, the number of clusters is 39.<br>
# 

# In[9]:


import numpy as np
from sklearn.cluster import KMeans

# set number of clusters = number of neighborhoods
kclusters = len(city['Neighborhood'].unique())

toronto_clustering = city.drop(columns={'PostalCode','Borough','Neighborhood'})

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(toronto_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_


# 
# <br>Constructing a new dataframe with Toronto neighborhood centers coordinates for mapping.

# In[10]:


neighborhoods = pd.DataFrame(columns=['Neighborhood','Cluster Labels','Latitude','Longitude']) 
names = city['Neighborhood'].unique()

for i in range(len(names)):
    one = names[i]
    two = kmeans.cluster_centers_[i][0]
    three = kmeans.cluster_centers_[i][1]
    four = kmeans.labels_[i]
    neighborhoods = neighborhoods.append({'Neighborhood':one,'Cluster Labels':four,'Latitude':two,'Longitude':three}, ignore_index=True)

neighborhoods.head()


# Creating a map with folium.

# In[11]:


# !conda install -c conda-forge folium=0.5.0 --yes
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors


# In[12]:


# create map
latitude = neighborhoods['Latitude'][3]
longitude = neighborhoods['Longitude'][3]
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=12)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(neighborhoods['Latitude'], neighborhoods['Longitude'], neighborhoods['Neighborhood'], neighborhoods['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# __A map with the Neighborhood centers for all 39 Neighborhoods in Toronto__

# ## <br><br>Assignment 3 - Financial Centers <a class="anchor" id="assignment3"></a>

# ### <br>Part 1 - Defining the problem <a class="anchor" id="part3.1"></a>

# World financial centers are some of the most wealthy regions. They are spacially very concentrated and a hot spot for luxury products. In the aftermath of the coronavirus pandemic, thousands of restaurants shut down for good, even in the busy financial centers. But ones downfall is the others chance and people always need places to go and eat. __New restaurants are needed__, so you take your stimulus check and open a restaurant in __Frankfurt am Main__ (short: FFM), Germany. But what kind of restaurant? What does the international financial elite consume (apart from cocaine)? Is there a common type of restaurant between those cities that promises success? Let's dive into the data and find out!

# ### <br>Part 2 - The Data <a class="anchor" id="part3.2"></a>

# We will compare data of the financial centers of the following Cities: __New York__, __Toronto__, __Tokyo__, __London__ and __Frankfurt__

# In[13]:


import numpy as np
import pandas as pd 
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

# !conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

#!conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab
import folium # map rendering library


# Adding _"&query=food"_ to the Foursquare API url to only get food-related venues. This way the search radius can be expanded without hitting the limit of 100 venues per city. The request and subsequent data frame building is looped over all 5 cities.

# In[14]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']

# list of cities and their financial centers as coordinates
cities = ["New York", "Toronto", "Tokyo", "London", "Frankfurt"]
latitudes = [40.7077426113486, 43.64803026178278, 35.679822325779995, 51.515259192193064, 50.11081120228777]
longitudes = [-74.01157215146745, -79.3816696346114, 139.77014548424125, -0.08039578093893954, 8.67324827528688]

# Foursquare API credentials
CLIENT_ID = 'T0ETMOBTMRGKDWDG5VBSUFRUVWYJDTLHMLZHN2APZ545NNGR' # your Foursquare ID
CLIENT_SECRET = 'T03SSSFDVCM1JFVTX4YJLGS2IJPYD2AMRNWAR4DAXF2RFMZA' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version
LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 10000 # define radius

# a list to store all data frames in
all_df = list()

# loop over all 5 cities
for i in range(len(cities)):
# create URL
    url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}&query=food'.format(
        CLIENT_ID, 
        CLIENT_SECRET, 
        VERSION, 
        latitudes[i], 
        longitudes[i], 
        radius, 
        LIMIT)

    results = requests.get(url).json()
    venues = results['response']['groups'][0]['items']
    nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
    filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
    nearby_venues = nearby_venues.loc[:, filtered_columns]

# filter the category for each row
    nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
    nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]
    nearby_venues['city'] = cities[i]
# add city name to column
    all_df.append(nearby_venues)
    
all_df[0].head()


# <br>Although bistros and cafes also serve food, for simplicity, we only use restaurant-type categories.

# In[15]:


# list restaurants data frames
restaurants = list()
for i in range(5):
    restaurants.append(all_df[i][all_df[i]['categories'].str.match(".*Restaurant*.")])
restaurants[1].head()


# Some restaurants don't have a specification for their kitchen. These will be treated as "miscellaneous" category.

# In[16]:


for i in range(5):
    print("Number of restaurants in",cities[i],":  ",len(restaurants[i]))
    print("Misc. Restaurants:", sum(restaurants[i]['categories'].str.match("Restaurant")) )


# In[17]:


# renaming "Restaurants" to "Miscellaneous Restaurant"
for i in range(5):
    restaurants[i].loc[:]['categories'] = restaurants[i].loc[:]['categories'].replace('Restaurant','Miscellaneous Restaurant')
restaurants[1].head()


# <br>Now let's map the data.

# In[18]:


map_all = folium.Map(location=[0, 0], zoom_start=2)

# add markers to map
for i in range(5):
    for lat, lng, name, category in zip(restaurants[i]['lat'], restaurants[i]['lng'], restaurants[i]['name'], restaurants[i]['categories']):
        label = '{}, {}'.format(name, category)
        label = folium.Popup(label, parse_html=True)
        folium.CircleMarker(
            [lat, lng],
            radius=5,
            popup=label,
            color='blue',
            fill=True,
            fill_color='#3186cc',
            fill_opacity=0.7,
            parse_html=False).add_to(map_all)  
    
map_all


# In[ ]:





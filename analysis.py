#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#reading datasets into dataframes
listings = pd.read_csv("listings.csv")
print(listings.head())
evictions = pd.read_csv("evictions.csv")
print(evictions.head())


'''we should try to analyze by borough bc there are way too many individual neighborhoods in nyc - both datasets have boroughs but we need to clean the data so that they align. so like number of evictions within each borough and see if the density of airbnbs in each borough goes up w eviction rates'''
#data wrangling
evictions = evictions[evictions['Residential/Commercial'] == 'Residential'].copy() #residential evictions only
evictions['BOROUGH'] = evictions['BOROUGH'].str.title()

listings.drop(columns=['price', 'license', 'host_profile_id']) #these columns are empty and/or irrelevant
listings['is_entire_home'] = listings['room_type'] == 'Entire home/apt'

eviction_counts = (evictions[evictions['Residential/Commercial'] == 'Residential'].groupby('BOROUGH').size().reset_index(name='eviction_count'))
listing_agg = (listings.groupby('neighbourhood_group').agg(total_listings = ('id', 'count'), entire_home_count = ('is_entire_home', 'sum')).reset_index())
listing_agg['entire_home_proportion'] = (listing_agg['entire_home_count'] / listing_agg['total_listings'])
merged = pd.merge(eviction_counts, listing_agg, left_on='BOROUGH', right_on='neighbourhood_group', how='inner') #merging dataset borough columns

#visualizations

#data analysis/statistical significance
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading datasets into dataframes
listings = pd.read_csv("listings.csv", index_col=0)
listings.drop(columns=['price'], inplace=True) #the entire column is blank for some reason so i think we should drop it
print(listings.head())
evictions = pd.read_csv("evictions.csv", index_col=0)
print(evictions.head())


'''we should try to analyze by borough - both datasets have boroughs but we need to clean the data so that they align. so like number of evictions within each borough and see if the density of airbnbs in each borough goes up w eviction rates'''
#calculations
brooklyn_count = listings['neighbourhood_group'].value_counts()['Brooklyn']
manhattan_count = listings['neighbourhood_group'].value_counts()['Manhattan']
queens_count = listings['neighbourhood_group'].value_counts()['Queens']
bronx_count = listings['neighbourhood_group'].value_counts()['Bronx']
staten_count = listings['neighbourhood_group'].value_counts()['Staten Island'] #or could just add new column in listings csv but idk it feels weird

#visualizations

#data analysis/statistical significance
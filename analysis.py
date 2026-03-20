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

#we should try to analyze by borough - both datasets have boroughs but we need to clean the data so that they align

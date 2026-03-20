#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading datasets into dataframes
listings = pd.read_csv("listings.csv", index_col=0)
print(listings.head())
evictions = pd.read_csv("evictions.csv", index_col=0)

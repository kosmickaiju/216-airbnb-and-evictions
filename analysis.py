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
fig, ax = plt.subplots(figsize=(8, 5))
sns.regplot(data=merged, x='entire_home_proportion', y='eviction_count', ax=ax, scatter_kws={'s': 100, 'zorder': 5, 'alpha': 0.8}, line_kws={'color': 'black', 'linestyle': '--', 'linewidth': 1.5})
for _, row in merged.iterrows():
    ax.annotate(row['BOROUGH'], xy=(row['entire_home_proportion'], row['eviction_count']), xytext=(6, 4), textcoords='offset points', fontsize=9)
ax.set_xlabel('Proportion of Entire-Home Airbnb Listings')
ax.set_ylabel('Total Eviction Count')
ax.set_title('Entire-Home Airbnb Proportion vs. Evictions by Borough')
plt.tight_layout()
plt.savefig('scatter_borough.png', dpi=150)

#data analysis/statistical significance
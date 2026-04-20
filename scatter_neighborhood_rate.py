import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

listings = pd.read_csv("data/listings.csv")
evictions = pd.read_csv("data/evictions.csv")
neighborhoods = gpd.read_file("data/nyc_neighborhoods.json").to_crs("EPSG:4326")

# filter out irrelevant columns and create mask that indicates listing is an entire home
listings = listings.drop(columns=["price", "license", "host_profile_id"])
listings["is_entire_home"] = listings["room_type"] == "Entire home/apt"

# convert listings dataframe to a geodataframe based on coordinates
listings_gdf = gpd.GeoDataFrame(
    listings.dropna(subset=["latitude", "longitude"]),
    geometry=gpd.points_from_xy(listings.dropna(subset=["latitude", "longitude"]).longitude,
                                listings.dropna(subset=["latitude", "longitude"]).latitude),
    crs="EPSG:4326"
)

# match listings to neighborhood geographies and count number of listings in each neighborhood
joined = gpd.sjoin(listings_gdf, neighborhoods[["ntaname", "geometry"]], how="left", predicate="within")
listing_counts = (joined.groupby("ntaname").agg(total_listings=("id", "count"), entire_home_count=("is_entire_home", "sum")).reset_index())
listing_counts["entire_home_proportion"] = listing_counts["entire_home_count"] / listing_counts["total_listings"]

# filter evictions dataframe to 2019 and only residential
evictions["year"] = pd.to_datetime(evictions["Executed Date"], errors="coerce").dt.year
evictions_2019 = evictions[(evictions["year"] == 2019) & (evictions["Residential/Commercial"] == "Residential")]
eviction_counts = evictions_2019.groupby("NTA").size().reset_index(name="eviction_count")

# read ACS 2019 data and NTA crosswalk table
acs = pd.read_csv("data/acs_2019_population/tract_population.csv", skiprows=1) # skiprows=1 to skip weird code headers
crosswalk = pd.read_excel("data/NTA_crosswalk.xlsx")

# decode GEOIDs into county and tract -- each set of numbers encodes a specific geography type
acs["county"] = acs["Geography"].str[11:14].astype(int)
acs["tract"] = acs["Geography"].str[14:20].astype(int)

# only keep counties in NYC
acs_nyc = acs[acs["county"].isin([5, 47, 61, 81, 85])].copy() 

# merge population data with crosswalk on county and tract
acs_crosswalk = acs_nyc.merge(
    crosswalk,
    left_on=["county", "tract"],
    right_on=["2010 Census Bureau FIPS County Code", "2010 Census Tract"],
    how="inner",
)

# aggregate population by neighborhood across all census tracts within a neighborhood
nta_pop = (
    acs_crosswalk.groupby("Neighborhood Tabulation Area (NTA) Name")["Estimate!!Total"]
    .sum()
    .reset_index()
)
nta_pop.columns = ["ntaname", "population"]

# fix name mismatch between crosswalk and GeoJSON
nta_pop["ntaname"] = nta_pop["ntaname"].str.replace("Flat Iron", "Flatiron", regex=False)

# merge everything
merged = listing_counts.merge(eviction_counts, left_on="ntaname", right_on="NTA", how="inner")
merged = merged[~merged["ntaname"].str.startswith("park-cemetery-etc")] # take out all park-cemetary neighborhoods
merged = merged.merge(nta_pop, on="ntaname", how="inner")

# calculate eviction rate
merged["eviction_rate"] = (merged["eviction_count"] / merged["population"]) * 1000

# plot
fig, ax = plt.subplots(figsize=(8, 5))
sns.regplot(
    data=merged,
    x="entire_home_proportion",
    y="eviction_rate",
    ax=ax,
    scatter_kws={"s": 40, "zorder": 5, "alpha": 0.8},
    line_kws={"color": "black", "linestyle": "--", "linewidth": 1.5},
)

ax.set_xlabel("Proportion of Entire-Home Airbnb Listings")
ax.set_ylabel("Eviction Rate (per 1000 residents)")
ax.set_title("Entire-Home Airbnb Proportion vs. Eviction Rate by Neighborhood")
plt.tight_layout()
plt.savefig("outputs/scatter_neighborhood_rate.png", dpi=150)
plt.show()

# statistical analysis: one-tailed test
'''
results: 
r: -0.3603
r^2: 0.1298
p-value: 0.0000

The negative r-value indicates a negative correlation which is the opposite of our hypothesis. 
r^2=0.13 indicates that approximately 13% of variance in eviction rates is explained by Airbnb proportion. 
p-value=0 suggests that the negative correlation between eviction rate and Airbnb proportion is highly statistically significant.
'''
r, p_two_tailed = stats.pearsonr(merged["entire_home_proportion"], merged["eviction_rate"])
p_one_tailed = p_two_tailed / 2  
r_squared = r ** 2

print(f"r: {r:.4f}")
print(f"r^2: {r_squared:.4f}")
print(f"p-value: {p_one_tailed:.4f}")



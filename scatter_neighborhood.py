import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

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

# merge
merged = pd.merge(listing_counts, eviction_counts, left_on="ntaname", right_on="NTA", how="inner")

# plot
fig, ax = plt.subplots(figsize=(8, 5))
sns.regplot(
    data=merged,
    x="entire_home_proportion",
    y="eviction_count",
    ax=ax,
    scatter_kws={"s": 40, "zorder": 5, "alpha": 0.8},
    line_kws={"color": "black", "linestyle": "--", "linewidth": 1.5},
)

ax.set_xlabel("Proportion of Entire-Home Airbnb Listings")
ax.set_ylabel("Total Eviction Count")
ax.set_title("Entire-Home Airbnb Proportion vs. Evictions by Neighborhood")
plt.tight_layout()
plt.savefig("outputs/scatter_neighborhood.png", dpi=150)
plt.show()

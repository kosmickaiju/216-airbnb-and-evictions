import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

neighborhoods = gpd.read_file("data/nyc_neighborhoods.json").to_crs("EPSG:4326")

###################### Airbnb Heatmap ########################
listings = pd.read_csv("data/listings.csv")

# convert listings dataframe to a geodataframe based on coordinates
listings_gdf = gpd.GeoDataFrame(
    listings.dropna(subset=["latitude", "longitude"]),
    geometry=gpd.points_from_xy(listings.dropna(subset=["latitude", "longitude"]).longitude,
                                listings.dropna(subset=["latitude", "longitude"]).latitude),
    crs="EPSG:4326"
)

# match listings to neighborhood geographies and count number of listings in each neighborhood
joined = gpd.sjoin(listings_gdf, neighborhoods[["ntaname", "geometry"]], how="left", predicate="within")
counts = joined.groupby("ntaname").size().reset_index(name="listing_count")

# merge on neighborhood
neighborhoods = neighborhoods.merge(counts, on="ntaname", how="left")
neighborhoods["listing_count"] = neighborhoods["listing_count"].fillna(0).astype(int)

# plot
fig, ax = plt.subplots(figsize=(10, 12))
neighborhoods.plot(
    column="listing_count",
    ax=ax,
    cmap="YlOrRd",
    linewidth=0.3,
    edgecolor="white",
    legend=True,
    legend_kwds={"label": "Number of Airbnb Listings", "shrink": 0.6},
)
ax.set_title("Airbnb Listings by NYC Neighborhood", fontsize=16)
ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/airbnb_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()

####################### Eviction Heatmap #########################
evictions = pd.read_csv("data/evictions.csv")

# filter to 2019 residential evictions
evictions["year"] = pd.to_datetime(evictions["Executed Date"], errors="coerce").dt.year
evictions_2019 = evictions[(evictions["year"] == 2019) & (evictions["Residential/Commercial"] == "Residential")]

# count evictions per neighborhood
counts = evictions_2019.groupby("NTA").size().reset_index(name="eviction_count")

# merge on neighborhood
neighborhoods = neighborhoods.merge(counts, left_on="ntaname", right_on="NTA", how="left")
neighborhoods["eviction_count"] = neighborhoods["eviction_count"].fillna(0).astype(int)

# plot
fig, ax = plt.subplots(figsize=(10, 12))
neighborhoods.plot(
    column="eviction_count",
    ax=ax,
    cmap="YlOrRd",
    linewidth=0.3,
    edgecolor="white",
    legend=True,
    legend_kwds={"label": "Number of Evictions (2019)", "shrink": 0.6},
)
ax.set_title("Residential Evictions by NYC Neighborhood (2019)", fontsize=16)
ax.axis("off")
plt.tight_layout()
plt.savefig("outputs/evictions_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
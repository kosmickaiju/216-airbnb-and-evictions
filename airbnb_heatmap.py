import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

listings = pd.read_csv("listings.csv")
neighborhoods = gpd.read_file("nyc_neighborhoods.json").to_crs("EPSG:4326")

# match listings to neighborhood geographies
listings_gdf = gpd.GeoDataFrame(
    listings.dropna(subset=["latitude", "longitude"]),
    geometry=gpd.points_from_xy(listings.dropna(subset=["latitude", "longitude"]).longitude,
                                listings.dropna(subset=["latitude", "longitude"]).latitude),
    crs="EPSG:4326"
)

# count number of listings in each neighborhood
joined = gpd.sjoin(listings_gdf, neighborhoods[["ntaname", "geometry"]], how="left", predicate="within")
counts = joined.groupby("ntaname").size().reset_index(name="listing_count")

neighborhoods = neighborhoods.merge(counts, on="ntaname", how="left")
neighborhoods["listing_count"] = neighborhoods["listing_count"].fillna(0).astype(int)


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
plt.savefig("airbnb_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import statsmodels.api as sm
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

# --- ACS: population and income aggregated to NTA via crosswalk ---
pop = pd.read_csv("data/acs_2019_population/tract_population.csv", skiprows=1)
inc = pd.read_csv("data/acs_2019_income/tract_income.csv", skiprows=1)
xwalk = pd.read_excel("data/NTA_crosswalk.xlsx")

for df in [pop, inc]:
    df["county"] = df["Geography"].str[11:14].astype(int)
    df["tract"] = df["Geography"].str[14:20].astype(int)

nyc_counties = [5, 47, 61, 81, 85]
pop_nyc = pop[pop["county"].isin(nyc_counties)].copy()
inc_nyc = inc[inc["county"].isin(nyc_counties)].copy()

inc_col = "Estimate!!Median household income in the past 12 months (in 2019 inflation-adjusted dollars)"
inc_nyc = inc_nyc[["county", "tract", inc_col]].copy()
inc_nyc[inc_col] = pd.to_numeric(inc_nyc[inc_col], errors="coerce")  # nulls encoded as non-numeric in ACS

# join population and income at tract level
tract_data = pop_nyc.merge(inc_nyc, on=["county", "tract"], how="inner")
tract_data = tract_data.merge(
    xwalk,
    left_on=["county", "tract"],
    right_on=["2010 Census Bureau FIPS County Code", "2010 Census Tract"],
    how="inner",
)
tract_data = tract_data.rename(columns={
    "Estimate!!Total": "population",
    inc_col: "median_income",
    "Neighborhood Tabulation Area (NTA) Name": "ntaname",
})

# population-weighted average income per NTA
tract_data["income_x_pop"] = tract_data["median_income"] * tract_data["population"]
nta_agg = (
    tract_data.groupby("ntaname")
    .agg(population=("population", "sum"), income_x_pop=("income_x_pop", "sum"))
    .reset_index()
)
nta_agg["median_income"] = nta_agg["income_x_pop"] / nta_agg["population"]
nta_agg = nta_agg.drop(columns="income_x_pop")
nta_agg["ntaname"] = nta_agg["ntaname"].str.replace("Flat Iron", "Flatiron", regex=False)

# --- merge everything ---
merged = listing_counts.merge(eviction_counts, left_on="ntaname", right_on="NTA", how="inner")
merged = merged[~merged["ntaname"].str.startswith("park-cemetery-etc")]
merged = merged.merge(nta_agg, on="ntaname", how="inner")
merged["eviction_rate"] = (merged["eviction_count"] / merged["population"]) * 1000
merged = merged.dropna(subset=["median_income"])

# --- plot: scatter colored by income ---
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(
    merged["entire_home_proportion"],
    merged["eviction_rate"],
    c=merged["median_income"],
    cmap="RdYlGn",
    s=40,
    alpha=0.8,
    zorder=5,
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("Median Household Income ($)")
ax.set_xlabel("Proportion of Entire-Home Airbnb Listings")
ax.set_ylabel("Eviction Rate (per 1,000 residents)")
ax.set_title("Entire-Home Airbnb Proportion vs. Eviction Rate\nColored by Median Household Income")
plt.tight_layout()
plt.savefig("outputs/scatter_income.png", dpi=150)
plt.show()

# --- multiple regression: eviction rate ~ airbnb proportion + income ---
'''
results:
Multiple Regression: Eviction Rate ~ Airbnb Proportion + Median Income
R²:                        0.4360
Airbnb proportion coef:    0.1118  (p = 0.8348)
Median income coef:        -0.000035  (p = 0.0000)

The multiple regression attempts to analyze how Airbnb proportion predicts eviction rates while accounting for household income. 
r^2=0.4360 indicates that about 44% of the variance in eviction rates is explained by income and Airbnbs. 
Airbnb r=0.1118 does indicate that higher Airbnb proportion does correlate with higher eviction rates. However, p=0.835 suggests that this 
relation is highly insignificant statistically.
Income r=-0.00004 indicates that higher income is correlated with lower eviction rates. p=0 suggests that this relation is highly 
statistically significant.
'''
X = sm.add_constant(merged[["entire_home_proportion", "median_income"]])
y = merged["eviction_rate"]
model = sm.OLS(y, X).fit()

print("Multiple Regression: Eviction Rate ~ Airbnb Proportion + Median Income")
print(f"R²:                        {model.rsquared:.4f}")
print(f"Airbnb proportion coef:    {model.params['entire_home_proportion']:.4f}  (p = {model.pvalues['entire_home_proportion']:.4f})")
print(f"Median income coef:        {model.params['median_income']:.6f}  (p = {model.pvalues['median_income']:.4f})")

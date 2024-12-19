import pandas as pd
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
import branca.colormap as cm
from esda.getisord import G_Local
from libpysal.weights import Queen

'''Python script used to make all the maps for the project.'''


def color_for_z(z):
    z = float(z)
    if z >= 2.56:
        return "#8B0000"  # deep red
    elif z >= 1.96:
        return "#FF4500"  # lighter red
    elif z <= -2.56:
        return "#00008B"  # deep blue
    elif z <= -1.96:
        return "#0000FF"  # lighter blue
    else:
        return "#FFFFFF"  # neutral


def plot_zscore_map(gdf, output_path):
    m = folium.Map(
        location=[31.0, -99.0], zoom_start=6, tiles="CartoDB positron"
    )
    geojson_data = gdf.__geo_interface__

    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            "fillColor": color_for_z(feature["properties"]["Z_score"]),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7,
        },
        highlight_function=lambda x: {
            "weight": 3,
            "color": "black",
            "fillOpacity": 0.9,
        },
        tooltip=GeoJsonTooltip(
            fields=["CNTY_NM", "Z_score"],
            aliases=["County:", "Z-Score:"],
            localize=True,
        ),
    ).add_to(m)

    m.save(output_path)


def plot_map(gdf, value_col, caption, output_path, max_value=None):
    min_value = 0
    if max_value is None:
        max_value = gdf[value_col].max()

    colorscale = cm.linear.YlOrRd_09.scale(min_value, max_value)

    m = folium.Map(
        location=[31.0, -99.0], zoom_start=6, tiles="CartoDB positron"
    )
    geojson_data = gdf.__geo_interface__

    folium.GeoJson(
        geojson_data,
        style_function=lambda feature: {
            "fillColor": colorscale(feature["properties"][value_col]),
            "color": "black",
            "weight": 1,
            "fillOpacity": 0.7,
        },
        highlight_function=lambda x: {
            "weight": 3,
            "color": "black",
            "fillOpacity": 0.9,
        },
        tooltip=GeoJsonTooltip(
            fields=["CNTY_NM", value_col],
            aliases=["County:", caption],
            localize=True,
        ),
    ).add_to(m)

    colorscale.caption = caption
    colorscale.add_to(m)

    m.save(output_path)


def compute_local_g(gdf, column):
    """Compute local G statistics for a given column in a GeoDataFrame."""
    y = gdf[column].values
    w = Queen.from_dataframe(gdf, use_index=False)
    w.transform = "r"
    lg = G_Local(y, w, permutations=999)
    gdf["G_i"] = lg.Gs
    gdf["Z_score"] = lg.Zs
    gdf["p_value"] = lg.p_sim
    return gdf


def load_fire_data(fire_data_path, shapefile_path, year=None):
    df = pd.read_csv(fire_data_path)
    df["FIPS"] = df["FIPS"].astype(str)

    if year is not None:
        # Filter data for the given year (assuming column name 'FIRE_YEAR')
        df = df[df["FIRE_YEAR"] == year]

    # Column names differ if year provided or not
    count_col = "total_fires_year" if year is not None else "total_fires"
    acres_col = "total_acres_year" if year is not None else "total_acres"

    fires_by_county = (
        df.groupby("FIPS", as_index=False)
        .agg(
            **{count_col: ("FOD_ID", "count"), acres_col: ("FIRE_SIZE", "sum")}
        )
        .fillna(0)
    )

    # Merge with county shapefile
    gdf = gpd.read_file(shapefile_path)
    merged_gdf = gdf.merge(
        fires_by_county, left_on="CNTY_FIPS", right_on="FIPS", how="left"
    )
    merged_gdf[count_col] = merged_gdf[count_col].fillna(0)
    merged_gdf[acres_col] = merged_gdf[acres_col].fillna(0)

    # Compute Local G stats
    merged_gdf = compute_local_g(merged_gdf, count_col)

    return merged_gdf


def load_drought_data(drought_data_path, shapefile_path, year=None):
    drought_df = pd.read_csv(drought_data_path)
    drought_df["FIPS"] = drought_df["FIPS"].astype(str)

    # Extract the year from the MapDate column
    drought_df["Year"] = drought_df["MapDate"].astype(str).str[:4].astype(int)

    if year is not None:
        drought_df = drought_df[drought_df["Year"] == year]
        val_col = "avg_drought_index_year"
    else:
        val_col = "avg_drought_index"

    drought_by_county = (
        drought_df.groupby("FIPS", as_index=False)
        .agg(**{val_col: ("DSCI", "mean")})
        .fillna(0)
    )

    gdf = gpd.read_file(shapefile_path)
    merged_gdf = gdf.merge(
        drought_by_county, left_on="CNTY_FIPS", right_on="FIPS", how="left"
    )
    merged_gdf[val_col] = merged_gdf[val_col].fillna(0)

    # Compute Local G stats
    merged_gdf = compute_local_g(merged_gdf, val_col)

    return merged_gdf


# Map creation functions
def create_total_fires_map(merged_gdf):
    print("Plotting total fires map...")
    plot_map(
        merged_gdf,
        value_col="total_fires",
        caption="Total Fires by County",
        output_path="maps/texas_choropleth_total_fires.html",
    )


def create_total_acres_map(merged_gdf):
    print("Plotting total acres map...")
    plot_map(
        merged_gdf,
        value_col="total_acres",
        caption="Total Acres Burned by County",
        output_path="maps/texas_choropleth_total_acres.html",
    )


def create_gi_map_for_fires(merged_gdf):
    print("Plot Local GI map (fires)")
    plot_zscore_map(
        merged_gdf,
        "maps/texas_choropleth_Gi.html",
    )


def create_drought_map(merged_gdf):
    val_col = (
        "avg_drought_index"
        if "avg_drought_index" in merged_gdf.columns
        else "avg_drought_index_year"
    )
    caption = "Average Drought Severity by County"
    if val_col == "avg_drought_index_year":
        caption = (
            "Average Drought Severity by County in Year"
        )

    print("Plotting drought map...")
    plot_map(
        merged_gdf,
        value_col=val_col,
        caption=caption,
        output_path="maps/texas_choropleth_drought.html",
    )


def create_gi_map_for_drought(merged_gdf):
    print("Plotting drought GI map...")
    plot_zscore_map(
        merged_gdf,
        "maps/texas_choropleth_Gi_drought.html",
    )


def create_total_fires_map_year(merged_gdf, year, max_value=None):
    print(f"Plotting total fires map for {year}...")
    plot_map(
        merged_gdf,
        value_col="total_fires_year",
        caption=f"Total Fires by County in {year}",
        output_path=f"maps/texas_choropleth_total_fires_{year}.html",
        max_value=max_value,
    )


def create_gi_map_for_fires_year(merged_gdf, year):
    print(f"Plot Local GI map (fires) for {year}")
    plot_zscore_map(
        merged_gdf,
        f"maps/texas_choropleth_Gi_{year}.html",
    )


def create_drought_map_year(merged_gdf, year, max_value=None):
    print(f"Plotting drought map for {year}...")
    plot_map(
        merged_gdf,
        value_col="avg_drought_index_year",
        caption=f"Average Drought Severity by County in {year}",
        output_path=f"maps/texas_choropleth_drought_{year}.html",
        max_value=max_value,
    )


def create_gi_map_for_drought_year(merged_gdf, year):
    print(f"Plotting drought GI map for {year}...")
    plot_zscore_map(
        merged_gdf,
        f"maps/texas_choropleth_Gi_drought_{year}.html",
    )


def main():
    shapefile_path = "dataset_gitignore/County_shape.geojson"
    fire_data_path = "dataset/texas_fires_drought.csv"
    drought_data_path = "dataset/drought_county_tx.csv"
    year = 2011

    # Fire data average all data
    fire_gdf = load_fire_data(fire_data_path, shapefile_path, year=None)
    create_total_fires_map(fire_gdf)
    create_total_acres_map(fire_gdf)
    create_gi_map_for_fires(fire_gdf)

    # Drought data average all data
    drought_gdf = load_drought_data(
        drought_data_path, shapefile_path, year=None
    )
    create_drought_map(drought_gdf)
    create_gi_map_for_drought(drought_gdf)

    # Fire data for a specific year 2011
    fire_gdf_2011 = load_fire_data(fire_data_path, shapefile_path, year=year)
    create_total_fires_map_year(fire_gdf_2011, year)
    create_gi_map_for_fires_year(fire_gdf_2011, year)

    # Drought data for a specific year 2011
    drought_gdf_2011 = load_drought_data(
        drought_data_path, shapefile_path, year=year
    )
    create_drought_map_year(drought_gdf_2011, year)
    create_gi_map_for_drought_year(drought_gdf_2011, year)

    # Fire data for a specific year 2010
    max_value_fires = fire_gdf_2011["total_fires_year"].max()
    fire_gdf_2010 = load_fire_data(fire_data_path, shapefile_path, year=2010)
    create_total_fires_map_year(fire_gdf_2010, 2010, max_value_fires)
    create_gi_map_for_fires_year(fire_gdf_2010, 2010)

    # Drought data for a specific year 2010
    max_value_drought = drought_gdf_2011["avg_drought_index_year"].max()
    drought_gdf_2010 = load_drought_data(
        drought_data_path, shapefile_path, year=2010
    )
    create_drought_map_year(drought_gdf_2010, 2010, max_value_drought)
    create_gi_map_for_drought_year(drought_gdf_2010, 2010)


if __name__ == "__main__":
    main()

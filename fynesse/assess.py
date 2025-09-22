from typing import Any, Union
import pandas as pd
import logging

from .config import *
from . import access

# Set up logging
logger = logging.getLogger(__name__)

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded.
How are missing values encoded, how are outliers encoded? What do columns represent,
makes rure they are correctly labeled. How is the data indexed. Crete visualisation
routines to assess the data (e.g. in bokeh). Ensure that date formats are correct
and correctly timezoned."""


def data() -> Union[pd.DataFrame, Any]:
    """
    Load the data from access and ensure missing values are correctly encoded as well as
    indices correct, column names informative, date and times correctly formatted.
    Return a structured data structure such as a data frame.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR DATA ASSESSMENT CODE:
       - Load data using the access module
       - Check for missing values and handle them appropriately
       - Validate data types and formats
       - Clean and prepare data for analysis

    2. ADD ERROR HANDLING:
       - Handle cases where access.data() returns None
       - Check for data quality issues
       - Validate data structure and content

    3. ADD BASIC LOGGING:
       - Log data quality issues found
       - Log cleaning operations performed
       - Log final data summary

    4. EXAMPLE IMPLEMENTATION:
       df = access.data()
       if df is None:
           print("Error: No data available from access module")
           return None

       print(f"Assessing data quality for {len(df)} rows...")
       # Your data assessment code here
       return df
    """
    logger.info("Starting data assessment")

    # Load data from access module
    df = access.data()

    # Check if data was loaded successfully
    if df is None:
        logger.error("No data available from access module")
        print("Error: Could not load data from access module")
        return None

    logger.info(f"Assessing data quality for {len(df)} rows, {len(df.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your data assessment code here

        # Example: Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.info(f"Found missing values: {missing_counts.to_dict()}")
            print(f"Missing values found: {missing_counts.sum()} total")

        # Example: Check data types
        logger.info(f"Data types: {df.dtypes.to_dict()}")

        # Example: Basic data cleaning (students should customize this)
        # Remove completely empty rows
        df_cleaned = df.dropna(how="all")
        if len(df_cleaned) < len(df):
            logger.info(f"Removed {len(df) - len(df_cleaned)} completely empty rows")

        logger.info(f"Data assessment completed. Final shape: {df_cleaned.shape}")
        return df_cleaned

    except Exception as e:
        logger.error(f"Error during data assessment: {e}")
        print(f"Error assessing data: {e}")
        return None


def query(data: Union[pd.DataFrame, Any]) -> str:
    """Request user input for some aspect of the data."""
    raise NotImplementedError


def view(data: Union[pd.DataFrame, Any]) -> None:
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError


def labelled(data: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError

import osmnx as ox
import matplotlib.pyplot as plt

def plot_city_map(place_name: str, lat: float, lon: float, box_km: float = 2.0):
    """
    Plot a map of a city with streets, buildings, and POIs within a bounding box.

    Args:
        place_name (str): City or area name.
        lat (float): Center latitude.
        lon (float): Center longitude.
        box_km (float): Size of the bounding box in kilometers.
    """
    delta = (box_km / 2) / 111  # ~1 degree ≈ 111 km

    north = lat + delta
    south = lat - delta
    east = lon + delta
    west = lon - delta

    bbox = (north, south, east, west)

    graph = ox.graph_from_bbox(bbox, network_type="all")

    try:
        area = ox.geocode_to_gdf(place_name)
    except Exception:
        area = None

    nodes, edges = ox.graph_to_gdfs(graph)

    try:
        buildings = ox.geometries_from_bbox(north, south, east, west, tags={"building": True})
    except Exception:
        buildings = None

    try:
        pois = ox.geometries_from_bbox(north, south, east, west, tags={"amenity": True})
    except Exception:
        pois = None

    fig, ax = plt.subplots(figsize=(8, 8))

    if area is not None:
        area.plot(ax=ax, color="tan", alpha=0.5)

    if buildings is not None and not buildings.empty:
        buildings.plot(ax=ax, facecolor="gray", edgecolor="gray", alpha=0.7)

    edges.plot(ax=ax, linewidth=1, edgecolor="black", alpha=0.3)
    nodes.plot(ax=ax, color="black", markersize=1, alpha=0.3)

    if pois is not None and not pois.empty:
        pois.plot(ax=ax, color="green", markersize=5, alpha=1)

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_title(place_name, fontsize=14)
    plt.show()

# Comprehensive county name normalization function
def normalize_county_names(name):
    """
    Normalize county names to match the canonical forms in geo-boundaries.
    This function standardizes various formats to match 'ELGEYO-MARAKWET' format.
    """
    if pd.isna(name):
        return name

    # Convert to uppercase and strip whitespace
    name_clean = str(name).strip().upper()

    # Handle specific cases that need standardization
    # Elgeyo/Marakwet variations -> ELGEYO-MARAKWET
    if 'ELGEYO' in name_clean and 'MARAKWET' in name_clean:
        return 'ELGEYO-MARAKWET'

    # Tharaka-Nithi variations -> THARAKA (to match canonical geo boundaries)
    if name_clean in ['THARAKA-NITHI', 'THARAKA NITHI']:
        return 'THARAKA'

    # Taita/Taveta variations -> TAITA TAVETA
    if 'TAITA' in name_clean and 'TAVETA' in name_clean:
        return 'TAITA TAVETA'

    # Remove City suffix from Nairobi
    if name_clean == 'NAIROBI CITY':
        return 'NAIROBI'

    # Replace forward slashes and multiple hyphens with single spaces
    name_clean = name_clean.replace('/', ' ').replace('-', ' ')

    # Standardize multiple spaces to single spaces
    name_clean = ' '.join(name_clean.split())

    return name_clean


import geopandas as gpd
import matplotlib.pyplot as plt

def plot_primary_schools(schools_file, geojson_url):
    """
    Plots primary schools (public vs private) on Kenyan county boundaries.

    Parameters:
    schools_file (str): URL or path to GeoJSON for schools
    geojson_url (str): URL or path to Kenyan county boundaries GeoJSON

    Returns:
    None (displays matplotlib plot)
    """
    # Load county boundaries
    kenya_gdf = gpd.read_file(geojson_url)
    
    # Load schools data
    schools_gdf = gpd.read_file(schools_file)
    
    # Reproject schools CRS to match counties
    schools_gdf = schools_gdf.to_crs(kenya_gdf.crs)
    
    # Distinguish by status
    public_schools = schools_gdf[schools_gdf["Status"] == "Public"]
    private_schools = schools_gdf[schools_gdf["Status"] == "Private"]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 12))
    kenya_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=0.8)
    public_schools.plot(ax=ax, color="blue", markersize=1, label="Public Schools")
    private_schools.plot(ax=ax, color="red", markersize=1, label="Private Schools")
    
    ax.set_title("Primary Schools in Kenya (Public vs Private)", fontsize=12)
    ax.legend()
    plt.show()

import geopandas as gpd
import matplotlib.pyplot as plt
import os
import requests
import zipfile

def plot_schools_on_county_boundaries(
    schools_zip_url="https://github.com/call493/MLFC/raw/main/schools.zip",
    local_zip_path="schools.zip",
    extracted_dir="schools_extracted",
    geojson_url="https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/KEN/ADM1/geoBoundaries-KEN-ADM1.geojson"
):
    # Load counties
    kenya_gdf = gpd.read_file(geojson_url)

    # Download the zip file with school shapefiles
    print(f"Downloading {schools_zip_url}...")
    response = requests.get(schools_zip_url)
    response.raise_for_status()
    with open(local_zip_path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded to {local_zip_path}")

    # Extract the zip file
    print(f"Extracting {local_zip_path}...")
    with zipfile.ZipFile(local_zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_dir)
    print(f"Extracted to {extracted_dir}")

    # Find the .shp file in the extracted directory
    shp_file = None
    for root, _, files in os.walk(extracted_dir):
        for file in files:
            if file.endswith(".shp"):
                shp_file = os.path.join(root, file)
                break
        if shp_file:
            break

    if shp_file:
        print(f"Shapefile found at: {shp_file}")
        all_schools_gdf = gpd.read_file(shp_file)
        all_schools_gdf = all_schools_gdf.to_crs(kenya_gdf.crs)

        primary_schools = all_schools_gdf[all_schools_gdf["LEVEL"].str.lower() == "primary"]
        secondary_schools = all_schools_gdf[all_schools_gdf["LEVEL"].str.lower() == "secondary"]

        fig, ax = plt.subplots(figsize=(12, 12))
        kenya_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=0.8)
        if not primary_schools.empty:
            primary_schools.plot(ax=ax, color="green", markersize=5, label="Primary Schools")
        if not secondary_schools.empty:
            secondary_schools.plot(ax=ax, color="blue", markersize=5, label="Secondary Schools")
        ax.set_title("Schools in Kenya (Primary vs Secondary)", fontsize=12)
        ax.legend()
        plt.show()
    else:
        print("Shapefile (.shp) not found in the extracted directory.")

# Example usage:
# plot_schools_on_county_boundaries()

import matplotlib.pyplot as plt

def plot_population_heatmap(
    kenya_gdf, 
    population_df, 
    county_col_shape="shapeName", 
    county_col_pop="County", 
    population_col="Total",
    title="2019 Kenya Population Distribution by County", 
    cmap="OrRd", 
    figsize=(12, 12)
):
    # Normalize county names for matching
    population_df[county_col_pop] = population_df[county_col_pop].str.strip().str.upper()
    kenya_gdf[county_col_shape] = kenya_gdf[county_col_shape].str.strip().str.upper()

    # Merge population with county boundaries
    merged = kenya_gdf.merge(
        population_df, 
        left_on=county_col_shape, 
        right_on=county_col_pop, 
        how="left"
    )

    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    merged.plot(
        column=population_col,
        cmap=cmap,
        linewidth=0.8,
        edgecolor="black",
        legend=True,
        ax=ax
    )

    ax.set_title(title, fontsize=16)
    ax.axis("off")
    plt.show()

# Example usage:
# plot_population_heatmap(kenya_gdf, population_df)

import folium
from folium.plugins import HeatMap

def plot_interactive_population_map(
    kenya_gdf,
    population_df,
    county_col_shape="shapeName",
    county_col_pop="County",
    population_col="Total",
    map_center=[0.0236, 37.9062],
    zoom_start=6,
    fill_color="OrRd",
    legend_name="Population (2019)",
    map_tiles="cartodbpositron"
):
    # Normalize county names for matching
    population_df[county_col_pop] = population_df[county_col_pop].str.strip().str.upper()
    kenya_gdf[county_col_shape] = kenya_gdf[county_col_shape].str.strip().str.upper()

    # Merge population with county boundaries
    merged = kenya_gdf.merge(
        population_df, 
        left_on=county_col_shape, 
        right_on=county_col_pop, 
        how="left"
    )

    geojson_data = merged.to_json()

    m = folium.Map(location=map_center, zoom_start=zoom_start, tiles=map_tiles)

    folium.Choropleth(
        geo_data=geojson_data,
        data=merged,
        columns=[county_col_shape, population_col],
        key_on=f"feature.properties.{county_col_shape}",
        fill_color=fill_color,
        fill_opacity=1,
        line_opacity=0.1,
        legend_name=legend_name
    ).add_to(m)

    folium.GeoJson(
        geojson_data,
        name="County Boundaries",
        tooltip=folium.GeoJsonTooltip(
            fields=[county_col_shape, population_col],
            aliases=["County:", "Population:"]
        )
    ).add_to(m)

    return m

# Example usage:
# m = plot_interactive_population_map(kenya_gdf, population_df)
# m  # Display the map in Jupyter/Colab

def check_county_mismatches(
    kenya_gdf,
    population_df,
    merged,
    county_col_shape="shapeName",
    county_col_pop="County",
    population_col="Total"
):
    # Get unique county sets
    pop_counties = set(population_df[county_col_pop].unique())
    geo_counties = set(kenya_gdf[county_col_shape].unique())
    
    print(f"\nPopulation counties: {len(pop_counties)}")
    print(f"Geographic counties: {len(geo_counties)}")
    print(f"\nCounties in population data but not in geographic data: {pop_counties - geo_counties}")
    print(f"Counties in geographic data but not in population data: {geo_counties - pop_counties}")

    # Identify missing population data
    missing = merged[merged[population_col].isna()][[county_col_shape]]
    print(f"\nCounties with missing population data: {len(missing)}")
    if len(missing) > 0:
        print(missing)

# Example usage:
# check_county_mismatches(kenya_gdf, population_df, merged)

import matplotlib.pyplot as plt

def plot_county_population_heatmap(
    merged_gdf, 
    population_col="Total", 
    title="2019 Kenya Population Distribution by County", 
    cmap="OrRd", 
    figsize=(12, 12)
):
    fig, ax = plt.subplots(figsize=figsize)
    merged_gdf.plot(
        column=population_col,
        cmap=cmap,
        linewidth=0.8,
        edgecolor="black",
        legend=True,
        ax=ax
    )
    ax.set_title(title, fontsize=16)
    ax.axis("off")
    plt.show()

# Example usage:
# plot_county_population_heatmap(merged)

import plotly.express as px

def plot_primary_vs_secondary_schools(
    all_schools_gdf, 
    county_col="County", 
    level_col="LEVEL", 
    primary_label="Primary", 
    secondary_label="Secondary",
    title="Primary vs Secondary Schools by County in Kenya"
):
    # Aggregate counts by county and school level
    counts = all_schools_gdf.groupby([county_col, level_col]).size().unstack(fill_value=0).reset_index()
    
    # Create scatter plot
    fig = px.scatter(
        counts,
        x=primary_label,
        y=secondary_label,
        text=county_col,
        hover_name=county_col,
        labels={primary_label: primary_label, secondary_label: secondary_label},
        title=title,
    )
    fig.update_traces(marker=dict(size=10, opacity=0.7))
    fig.update_layout(showlegend=False)
    fig.show()

# Example usage:
# plot_primary_vs_secondary_schools(all_schools_gdf)

import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import zipfile
import requests
import os

def plot_schools_choropleth_map(
    schools_zip_url="https://github.com/call493/MLFC/raw/main/schools.zip",
    local_zip_path="schools.zip",
    extracted_dir="schools_extracted",
    geojson_url="https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/KEN/ADM1/geoBoundaries-KEN-ADM1.geojson",
    map_center=[0.2, 37.5],
    map_zoom=6,
    html_file="schools_counties.html",
    normalize_func=None
):
    # Download and extract the schools shapefile
    response = requests.get(schools_zip_url)
    with open(local_zip_path, "wb") as f:
        f.write(response.content)
    with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_dir)

    # Find the .shp file
    shp_file = None
    for root, _, files in os.walk(extracted_dir):
        for file in files:
            if file.endswith(".shp"):
                shp_file = os.path.join(root, file)
                break
        if shp_file:
            break

    # Load GeoDataFrames
    all_schools_gdf = gpd.read_file(shp_file)
    kenya_gdf = gpd.read_file(geojson_url)
    all_schools_gdf = all_schools_gdf.to_crs(kenya_gdf.crs)

    # Normalize county names
    if normalize_func is not None:
        all_schools_gdf['County'] = all_schools_gdf['County'].apply(normalize_func)
        kenya_gdf['shapeName'] = kenya_gdf['shapeName'].apply(normalize_func)
    else:
        all_schools_gdf['County'] = all_schools_gdf['County'].str.strip().str.upper()
        kenya_gdf['shapeName'] = kenya_gdf['shapeName'].str.strip().str.upper()

    # Aggregate school counts
    school_counts = all_schools_gdf.groupby(['County', 'LEVEL']).size().unstack(fill_value=0)

    # Merge with county boundaries
    kenya_gdf = kenya_gdf.merge(school_counts, how='left', left_on='shapeName', right_index=True)
    kenya_gdf["Primary"] = kenya_gdf.get("Primary", 0).fillna(0).astype(int)
    kenya_gdf["Secondary"] = kenya_gdf.get("Secondary", 0).fillna(0).astype(int)

    # Create Folium map
    m = folium.Map(location=map_center, zoom_start=map_zoom)

    folium.Choropleth(
        geo_data=kenya_gdf,
        name='Primary Schools',
        data=kenya_gdf,
        columns=['shapeName', 'Primary'],
        key_on='feature.properties.shapeName',
        fill_color='Greens',
        fill_opacity=0.6,
        line_opacity=0.4,
        legend_name='Number of Primary Schools'
    ).add_to(m)

    folium.Choropleth(
        geo_data=kenya_gdf,
        name='Secondary Schools',
        data=kenya_gdf,
        columns=['shapeName', 'Secondary'],
        key_on='feature.properties.shapeName',
        fill_color='Blues',
        fill_opacity=0.4,
        line_opacity=0.2,
        legend_name='Number of Secondary Schools'
    ).add_to(m)

    folium.GeoJson(
        kenya_gdf,
        name="Counties",
        style_function=lambda x: {'fillColor': '#00000000', 'color': 'black', 'weight': 1},
        tooltip=folium.GeoJsonTooltip(
            fields=['shapeName', 'Primary', 'Secondary'],
            aliases=['County', 'Primary Schools', 'Secondary Schools']),
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(html_file)
    return m

# Example usage:
# from fynesse.assess import normalize_county_names
# m = plot_schools_choropleth_map(normalize_func=normalize_county_names)
# m  # displays the map in Colab/Jupyter



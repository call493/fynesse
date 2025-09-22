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

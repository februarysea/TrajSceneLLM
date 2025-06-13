import osmnx as ox
import geopandas as gpd
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _fetch_features(place_name, tags, bbox=None):
    """
    Fetches features from OSM, with a bounding box fallback.
    """
    try:
        logging.info(f"Attempting to download features with tags {tags} for '{place_name}'...")
        gdf = ox.features_from_place(place_name, tags)
        logging.info(f"Successfully downloaded {len(gdf)} features for '{place_name}'.")
        return gdf
    except Exception as e:
        logging.warning(f"Could not download features for '{place_name}': {e}.")
        if bbox:
            logging.info(f"Attempting to download features using bounding box: {bbox}")
            try:
                north, south, east, west = bbox['north'], bbox['south'], bbox['east'], bbox['west']
                gdf = ox.features_from_bbox(north, south, east, west, tags)
                logging.info(f"Successfully downloaded {len(gdf)} features using bounding box.")
                return gdf
            except Exception as e2:
                logging.error(f"Failed to download features using bounding box: {e2}")
    return gpd.GeoDataFrame()


def download_road_network(config):
    """
    Downloads the road network for the specified place and saves it as a GraphML file.
    """
    processed_data_dir = Path(config['paths']['processed_data'])
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    road_network_path = processed_data_dir / config['paths']['road_network']
    place_name = config['osm']['place_name']
    
    if road_network_path.exists():
        logging.info(f"Road network already exists at {road_network_path}. Skipping download.")
        return

    logging.info(f"Downloading road network for '{place_name}'...")
    try:
        G = ox.graph_from_place(place_name, network_type="drive_service")
        ox.save_graphml(G, road_network_path)
        logging.info(f"Road network saved to {road_network_path}")
    except Exception as e:
        logging.error(f"Could not download road network for '{place_name}': {e}")

def download_poi_data(config):
    """
    Downloads and processes Points of Interest (POI) like subway and bus data.
    """
    place_name = config['osm']['place_name']
    bbox = config['osm']['bbox']
    processed_data_dir = Path(config['paths']['processed_data'])
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    # --- Download Subway Lines ---
    subway_lines_path = processed_data_dir / config['paths']['subway_lines']
    if not subway_lines_path.exists():
        logging.info("--- Downloading Subway Lines ---")
        subway_tags = config['osm']['subway_lines_tags']
        subway_lines = _fetch_features(place_name, subway_tags, bbox)
        if not subway_lines.empty:
            subway_lines.to_file(subway_lines_path, driver="GeoJSON")
            logging.info(f"Saved {len(subway_lines)} subway lines to {subway_lines_path}")
    else:
        logging.info(f"Subway lines already exist at {subway_lines_path}. Skipping.")

    # --- Download Subway Stations ---
    subway_stations_path = processed_data_dir / config['paths']['subway_stations']
    if not subway_stations_path.exists():
        logging.info("--- Downloading Subway Stations ---")
        station_tags = config['osm']['subway_stations_tags']
        subway_stations = _fetch_features(place_name, station_tags, bbox)
        if not subway_stations.empty:
            subway_stations.to_file(subway_stations_path, driver="GeoJSON")
            logging.info(f"Saved {len(subway_stations)} subway stations to {subway_stations_path}")
    else:
        logging.info(f"Subway stations already exist at {subway_stations_path}. Skipping.")
        
    # --- Download Bus Stops ---
    bus_stops_path = processed_data_dir / config['paths']['bus_stops']
    if not bus_stops_path.exists():
        logging.info("--- Downloading Bus Stops ---")
        bus_stop_gdfs = []
        for tags in config['osm']['bus_stops_queries']:
            gdf = _fetch_features(place_name, tags, bbox)
            if not gdf.empty:
                bus_stop_gdfs.append(gdf)
        
        if bus_stop_gdfs:
            bus_stops = gpd.GeoDataFrame(pd.concat(bus_stop_gdfs, ignore_index=True)).drop_duplicates(subset='geometry')
            bus_stops.to_file(bus_stops_path, driver="GeoJSON")
            logging.info(f"Saved {len(bus_stops)} unique bus stops to {bus_stops_path}")
    else:
        logging.info(f"Bus stops already exist at {bus_stops_path}. Skipping.")

    # --- Download Bus Routes ---
    bus_routes_path = processed_data_dir / config['paths']['bus_routes']
    if not bus_routes_path.exists():
        logging.info("--- Downloading Bus Routes ---")
        bus_route_gdfs = []
        for tags in config['osm']['bus_routes_queries']:
            gdf = _fetch_features(place_name, tags, bbox)
            if not gdf.empty:
                bus_route_gdfs.append(gdf)
        
        if bus_route_gdfs:
            bus_routes = gpd.GeoDataFrame(pd.concat(bus_route_gdfs, ignore_index=True)).drop_duplicates(subset='geometry')
            bus_routes.to_file(bus_routes_path, driver="GeoJSON")
            logging.info(f"Saved {len(bus_routes)} unique bus routes to {bus_routes_path}")
    else:
        logging.info(f"Bus routes already exist at {bus_routes_path}. Skipping.")


def run(config):
    """Runs the full OSM data download pipeline."""
    ox.settings.use_cache = True
    ox.settings.log_console = False # Set to False to avoid verbose osmnx output
    
    download_road_network(config)
    download_poi_data(config) 
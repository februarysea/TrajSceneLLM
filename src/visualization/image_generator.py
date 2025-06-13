import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from pathlib import Path
from tqdm import tqdm
import logging
import gc
import osmnx as ox

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _load_data(config):
    """Loads all necessary data for plotting."""
    processed_dir = Path(config['paths']['processed_data'])
    
    # Load trajectories
    traj_path = processed_dir / config['paths']['cleaned_trajectories']
    if not traj_path.exists():
        logging.error(f"Trajectory data not found at {traj_path}")
        return None, None, None, None
    
    df = pd.read_parquet(traj_path)
    gdf_points = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude), crs="EPSG:4326"
    )

    # Load road network
    roads_path = processed_dir / config['paths']['road_network']
    if not roads_path.exists():
        logging.error(f"Road network not found at {roads_path}")
        return None, None, None, None
    
    try:
        logging.info(f"Loading road network from {roads_path}...")
        G = ox.load_graphml(roads_path)
        # The 'edges' GeoDataFrame is what we need for plotting the road lines
        _, gdf_roads = ox.graph_to_gdfs(G)
        logging.info("Road network loaded and converted to GeoDataFrame.")
    except Exception as e:
        logging.error(f"Failed to load or process GraphML file {roads_path}: {e}")
        return None, None, None, None

    # Load subway lines
    subway_lines_path = processed_dir / config['paths']['subway_lines']
    subway_lines = gpd.read_file(subway_lines_path) if subway_lines_path.exists() else None
    if subway_lines is None:
        logging.warning("Subway lines data not found. Proceeding without it.")

    # Load bus stops
    bus_stops_path = processed_dir / config['paths']['bus_stops']
    bus_stops = gpd.read_file(bus_stops_path) if bus_stops_path.exists() else None
    if bus_stops is None:
        logging.warning("Bus stops data not found. Proceeding without it.")
        
    return gdf_points, gdf_roads, subway_lines, bus_stops

def _plot_single_trajectory(gdf_single_traj, gdf_roads, subway_lines, bus_stops, output_path, vis_config):
    """Plots and saves a single trajectory."""
    # Define plot bounds with padding
    minx, miny, maxx, maxy = gdf_single_traj.total_bounds
    padding = vis_config['padding']
    bounds = [minx - padding, miny - padding, maxx + padding, maxy + padding]

    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Clip and plot roads
    roads_in_range = gdf_roads.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
    roads_in_range.plot(ax=ax, color=vis_config['colors']['road'], linewidth=vis_config['linewidth']['roads'], alpha=vis_config['alpha']['roads'], zorder=1)
    
    legend_handles = [mpatches.Patch(color=vis_config['colors']['road'], label="Road Network")]

    # Clip and plot subway lines
    if subway_lines is not None:
        subway_in_range = subway_lines.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        if not subway_in_range.empty:
            subway_in_range.plot(ax=ax, color=vis_config['colors']['subway'], linewidth=vis_config['linewidth']['subway'], alpha=vis_config['alpha']['subway'], zorder=2)
            legend_handles.append(mpatches.Patch(color=vis_config['colors']['subway'], label='Subway Lines'))

    # Clip and plot bus stops
    if bus_stops is not None:
        bus_stops_in_range = bus_stops.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
        if not bus_stops_in_range.empty:
            bus_stops_in_range.plot(ax=ax, color=vis_config['colors']['bus_stop'], marker='o', 
                                    markersize=vis_config['markersize']['bus_stops'], alpha=vis_config['alpha']['bus_stops'], zorder=3)
            legend_handles.append(mpatches.Patch(color=vis_config['colors']['bus_stop'], label='Bus Stops'))

    # Plot trajectory
    gdf_single_traj.plot(ax=ax, color=vis_config['colors']['trajectory'], markersize=vis_config['markersize']['trajectory_points'], 
                         alpha=vis_config['alpha']['trajectory_points'], zorder=4)
    legend_handles.append(mpatches.Patch(color=vis_config['colors']['trajectory'], label="Trajectory"))

    # Configure plot appearance
    ax.set_facecolor('white')
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])

    plt.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=len(legend_handles), frameon=True)
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_path, dpi=vis_config['image_dpi'], bbox_inches='tight')
    plt.close(fig)
    gc.collect()

def run(config):
    """
    Generates images for each trajectory.
    """
    logging.info("--- Starting Image Generation ---")
    
    images_dir = Path(config['paths']['images'])
    images_dir.mkdir(parents=True, exist_ok=True)
    vis_config = config['visualization']
    
    logging.info("Loading data...")
    gdf_points, gdf_roads, subway_lines, bus_stops = _load_data(config)
    
    if gdf_points is None:
        logging.error("Could not load trajectory data. Aborting image generation.")
        return

    unique_ids = gdf_points["Unique_ID"].unique()
    logging.info(f"Found {len(unique_ids)} unique trajectories to plot.")

    for selected_id in tqdm(unique_ids, desc="Generating Images"):
        gdf_single_traj = gdf_points[gdf_points["Unique_ID"] == selected_id]
        transport_mode = gdf_single_traj['Transport Mode'].iloc[0] if 'Transport Mode' in gdf_single_traj.columns else 'unknown'
        
        output_filename = f"trajectory_{selected_id}_{transport_mode}.png"
        output_path = images_dir / output_filename

        if output_path.exists():
            continue

        try:
            _plot_single_trajectory(gdf_single_traj, gdf_roads, subway_lines, bus_stops, output_path, vis_config)
        except Exception as e:
            logging.error(f"Failed to plot trajectory {selected_id}: {e}")
            plt.close('all') # Close any lingering plots on error
            gc.collect()

    logging.info("--- Image Generation Complete ---") 
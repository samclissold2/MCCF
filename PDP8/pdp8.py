import sys, os, tempfile, zipfile, requests, time
import pandas    as pd
import numpy     as np
import geopandas as gpd
import matplotlib.pyplot as plt
from   matplotlib.colors import LinearSegmentedColormap
import contextily as ctx
import tempfile
import os
import zipfile
import requests
import shutil
import folium
import webbrowser
import os

# Create a temporary directory to store downloaded files
temp_dir = tempfile.mkdtemp()
print(f"Created temporary directory: {temp_dir}")

# GADM data for Vietnam
gadm_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_VNM_1.json.zip"
zip_path = os.path.join(temp_dir, "vietnam_provinces.zip")
json_path = os.path.join(temp_dir, "gadm41_VNM_1.json")


# Read the Excel file with both worksheets
file_path = r"C:\Users\SamClissold\MCCF\PDP8\PDP8 project data (1).xlsx"
solar_df = pd.read_excel(file_path, sheet_name='solar').set_index("Project")

# wind_df = pd.read_excel(file_path, sheet_name='Onshore')

solar_df = solar_df.reset_index()



# Create a temporary directory to store downloaded files
temp_dir = tempfile.mkdtemp()
print(f"Created temporary directory: {temp_dir}")

# --- GADM Data for Vietnam - LOCAL FILE ---
# Download gadm41_VNM_1.json.zip from
# https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_VNM_1.json.zip
# Extract gadm41_VNM_1.json and place it in a known directory
local_gadm_json_path = "gadm41_VNM_1.json"  # <--- EDIT THIS PATH if needed

# Check if the local file exists
if not os.path.exists(local_gadm_json_path):
    print(f"WARNING: Local GADM file not found at: {local_gadm_json_path}")
    print("Trying to download from server...")
    
    # GADM data for Vietnam
    gadm_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_VNM_1.json.zip"
    zip_path_gadm = os.path.join(temp_dir, "vietnam_provinces.zip")
    json_path_gadm = os.path.join(temp_dir, "gadm41_VNM_1.json")
    
    try:
        print(f"Downloading GADM data from {gadm_url}...")
        response_gadm = requests.get(gadm_url, timeout=60)
        response_gadm.raise_for_status()
        with open(zip_path_gadm, 'wb') as f:
            f.write(response_gadm.content)
        print(f"Extracting GADM data to {temp_dir}...")
        with zipfile.ZipFile(zip_path_gadm, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        print("GADM data downloaded and extracted.")
        json_path_gadm = os.path.join(temp_dir, "gadm41_VNM_1.json")
    except Exception as e:
        print(f"Error downloading GADM data: {e}")
        print("Please download the file manually and update the path.")
        sys.exit(1)
else:
    json_path_gadm = local_gadm_json_path
    print(f"Using local GADM data from: {json_path_gadm}")

# --- Load Solar Data ---
file_path = r"C:\Users\SamClissold\MCCF\PDP8\PDP8 project data (1).xlsx"
try:
    solar_df = pd.read_excel(file_path, sheet_name='solar')
    print("Solar data loaded successfully.")
except Exception as e:
    print(f"Error loading solar data: {e}")
    sys.exit(1)

# Convert capacity column to numeric
if 'Expected capacity (MW)' in solar_df.columns:
    solar_df['Expected capacity (MW)'] = pd.to_numeric(solar_df['Expected capacity (MW)'], errors='coerce')
else:
    print("Error: 'Expected capacity (MW)' column not found in solar data.")
    sys.exit(1)

# Print data summary
print(f"\nSolar data summary:")
print(f"Total rows: {len(solar_df)}")
print(f"Rows with valid Expected capacity (MW): {solar_df['Expected capacity (MW)'].notna().sum()}")

# Print unique values in timeline columns to debug
if '2025-2030' in solar_df.columns:
    print(f"Unique values in '2025-2030' column: {solar_df['2025-2030'].unique()}")
else:
    print("Warning: '2025-2030' column not found")
    
if '2030-2035' in solar_df.columns:
    print(f"Unique values in '2030-2035' column: {solar_df['2030-2035'].unique()}")
else:
    print("Warning: '2030-2035' column not found")

# Read Vietnam GeoJSON
vietnam = gpd.read_file(json_path_gadm)

# --- Visualization Setup ---
fig, ax = plt.subplots(1, 1, figsize=(15, 13))
plt.subplots_adjust(bottom=0.2)  # Adjust bottom margin for legends

# Set up the base map with province boundaries
vietnam.plot(ax=ax, color='lightgray', alpha=0.3, zorder=1)
vietnam.boundary.plot(ax=ax, linewidth=0.5, color='black', zorder=2)

# Set the axis limits to Vietnam's boundaries
vietnam_bounds = vietnam.total_bounds
buffer = 0.5  # degrees
ax.set_xlim([vietnam_bounds[0] - buffer, vietnam_bounds[2] + buffer])
ax.set_ylim([vietnam_bounds[1] - buffer, vietnam_bounds[3] + buffer])

# Convert solar_df to GeoDataFrame
gdf_projects = gpd.GeoDataFrame(
    solar_df,
    geometry=gpd.points_from_xy(solar_df.Latitude, solar_df.Longitude),  # Swapped as per previous discussions
    crs="EPSG:4326"
)

if vietnam.crs != gdf_projects.crs:
    gdf_projects = gdf_projects.to_crs(vietnam.crs)

# Create separate dataframes for early period (circles) and later period (squares)
early_period_projects = gdf_projects[
    (gdf_projects['2025-2030'] == '2025-2030') & 
    gdf_projects['Expected capacity (MW)'].notna()
].copy()

later_period_projects = gdf_projects[
    (gdf_projects['2030-2035'] == '2031-2035') & 
    gdf_projects['Expected capacity (MW)'].notna()
].copy()

print(f"Projects for 2025-2030 period: {len(early_period_projects)}")
print(f"Projects for 2031-2035 period: {len(later_period_projects)}")

# Define marker size scaling function
def scale_marker_size(capacity, min_cap, max_cap, min_size=20, max_size=1000):
    if pd.isna(capacity) or min_cap == max_cap:
        return min_size
    normalized_cap = (capacity - min_cap) / (max_cap - min_cap)
    return min_size + normalized_cap * (max_size - min_size)

# Get capacity range for all projects to ensure consistent scaling
all_valid_projects = gdf_projects.dropna(subset=['Expected capacity (MW)', 'geometry'])
if not all_valid_projects.empty:
    min_capacity = all_valid_projects['Expected capacity (MW)'].min()
    max_capacity = all_valid_projects['Expected capacity (MW)'].max()
    print(f"Capacity range: {min_capacity} to {max_capacity} MW")
else:
    min_capacity, max_capacity = 0, 1
    print("Warning: No valid projects found")

# Create colormaps
colors_solar = ['#FFEDA0', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026']
cmap_solar = LinearSegmentedColormap.from_list('solar_cmap', colors_solar)

# Plot 2025-2030 projects with circular markers
if not early_period_projects.empty:
    scatter_early = ax.scatter(
        early_period_projects.geometry.x,
        early_period_projects.geometry.y,
        s=early_period_projects['Expected capacity (MW)'].apply(
            lambda x: scale_marker_size(x, min_capacity, max_capacity)
        ),
        c=early_period_projects['Expected capacity (MW)'],
        cmap=cmap_solar,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5,
        marker='o',  # Circle markers for early period
        zorder=3,
        label='2025-2030 Period'
    )
    print(f"Plotted {len(early_period_projects)} projects for 2025-2030 period (circles)")

# Plot 2031-2035 projects with square markers
if not later_period_projects.empty:
    scatter_later = ax.scatter(
        later_period_projects.geometry.x,
        later_period_projects.geometry.y,
        s=later_period_projects['Expected capacity (MW)'].apply(
            lambda x: scale_marker_size(x, min_capacity, max_capacity)
        ),
        c=later_period_projects['Expected capacity (MW)'],
        cmap=cmap_solar,
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5,
        marker='s',  # Square markers for later period
        zorder=4,
        label='2031-2035 Period'
    )
    print(f"Plotted {len(later_period_projects)} projects for 2031-2035 period (squares)")

# Add colorbar (same for both since they use the same capacity values)
if not all_valid_projects.empty:
    cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    scatter_for_cbar = scatter_early if not early_period_projects.empty else scatter_later
    if 'scatter_for_cbar' in locals():
        cbar = plt.colorbar(scatter_for_cbar, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('Expected Capacity (MW)', fontsize=10)

# Add marker type legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', 
               markersize=10, label='2025-2030 Period (Circles)'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='grey', 
               markersize=10, label='2031-2035 Period (Squares)')
]

# Add marker size legend
capacity_examples = np.linspace(min_capacity, max_capacity, num=3)
for capacity in capacity_examples:
    size = scale_marker_size(capacity, min_capacity, max_capacity, min_size=20, max_size=1000)
    size_factor = np.sqrt(size) / 2  # Adjust marker size for legend display
    legend_elements.append(
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='grey',
                  markersize=size_factor, label=f'{capacity:.0f} MW')
    )

# Add the legend at the bottom of the plot
ax.legend(handles=legend_elements, loc='lower center', 
          bbox_to_anchor=(0.5, -0.15), ncol=len(legend_elements), frameon=True, fontsize=9)

# Add basemap
try:
    ctx.add_basemap(ax, crs=vietnam.crs.to_string(), source=ctx.providers.CartoDB.Positron, 
                    zoom=7, attribution_size=8)
except Exception as e:
    print(f"Could not add basemap: {e}")

# Title and remove axis lines/ticks
ax.set_title('Solar Power Projects in Vietnam (PDP8)', fontsize=15, pad=20)
ax.set_axis_off()

# Save and show the map
plt.savefig('vietnam_solar_projects_by_period.png', dpi=300, bbox_inches='tight')
print("\nSolar map saved as 'vietnam_solar_projects_by_period.png'")
plt.show()

# Create the Folium map
m = folium.Map([16,107], zoom_start=5, tiles="CartoDB.Positron")
for _, row in solar_df.iterrows():
    folium.CircleMarker(
        location=[row.Latitude, row.Longitude],
        radius=5 + row["Expected capacity (MW)"]**0.5,
        popup=f"{row.Project}: {row['Expected capacity (MW)']} MW",
        color="orange", fill=True
    ).add_to(m)

# Save the map to a file
output_dir = os.path.dirname(os.path.abspath(__file__))
map_file = os.path.join(output_dir, "solar_map.html")
m.save(map_file)
print(f"Map saved to: {map_file}")

# Try multiple approaches to open the browser
try:
    # Approach 1: Use webbrowser module with file:// protocol
    full_path = os.path.abspath(map_file)
    file_url = f"file://{full_path}"
    print(f"Attempting to open: {file_url}")
    
    # Try with the default browser
    import webbrowser
    if webbrowser.open(file_url):
        print("Browser opened successfully using webbrowser.open()")
    else:
        # Try with a specific browser
        for browser in ['chrome', 'firefox', 'safari', 'edge']:
            try:
                browser_controller = webbrowser.get(browser)
                browser_controller.open(file_url)
                print(f"Browser opened successfully using {browser}")
                break
            except:
                continue
        
        # Approach 2: Use os-specific commands as fallback
        import platform
        import subprocess
        
        system = platform.system()
        if system == 'Darwin':  # macOS
            subprocess.run(['open', map_file], check=False)
            print("Tried opening with macOS 'open' command")
        elif system == 'Windows':
            subprocess.run(['start', map_file], shell=True, check=False)
            print("Tried opening with Windows 'start' command")
        elif system == 'Linux':
            subprocess.run(['xdg-open', map_file], check=False)
            print("Tried opening with Linux 'xdg-open' command")
        
        # Approach 3: Print instructions for manual opening
        print("\nIf the map didn't open automatically, please:")
        print(f"1. Open your web browser")
        print(f"2. Press Ctrl+O (or Cmd+O on Mac)")
        print(f"3. Navigate to: {map_file}")
        print(f"4. Or copy and paste this URL in your browser: {file_url}")

except Exception as e:
    print(f"Error attempting to open browser: {e}")
    print(f"Please open the file manually at: {map_file}")

# Clean up
try:
    shutil.rmtree(temp_dir)
    print(f"Temporary directory {temp_dir} removed.")
except Exception as e:
    print(f"Error removing temporary directory {temp_dir}: {e}")


# Create a temporary directory to store downloaded files
temp_dir = tempfile.mkdtemp()
print(f"Created temporary directory: {temp_dir}")

# GADM data for Vietnam
gadm_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_VNM_1.json.zip"
zip_path = os.path.join(temp_dir, "vietnam_provinces.zip")
json_path = os.path.join(temp_dir, "gadm41_VNM_1.json")


# Read the Excel file with both worksheets
file_path = r"C:\Users\SamClissold\MCCF\PDP8\PDP8 project data (1).xlsx"
wind_df = pd.read_excel(file_path, sheet_name='onshore').set_index("Project")

# wind_df = pd.read_excel(file_path, sheet_name='Onshore')

wind_df = wind_df.reset_index()



# Create a temporary directory to store downloaded files
temp_dir = tempfile.mkdtemp()
print(f"Created temporary directory: {temp_dir}")

# GADM data for Vietnam
gadm_url = "https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_VNM_1.json.zip"
zip_path_gadm = os.path.join(temp_dir, "vietnam_provinces.zip")
json_path_gadm = os.path.join(temp_dir, "gadm41_VNM_1.json")

# Download and extract GADM data if not already present
if not os.path.exists(json_path_gadm):
    print(f"Downloading GADM data from {gadm_url}...")
    response_gadm = requests.get(gadm_url)
    with open(zip_path_gadm, 'wb') as f:
        f.write(response_gadm.content)
    print(f"Extracting GADM data to {temp_dir}...")
    with zipfile.ZipFile(zip_path_gadm, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    print("GADM data downloaded and extracted.")
else:
    print("GADM data already found in temporary directory.")

# Read the Excel file
file_path = r"C:\Users\SamClissold\MCCF\PDP8\PDP8 project data (1).xlsx" # Keep your actual file path

# --- Load Wind Data ---
try:
    wind_df = pd.read_excel(file_path, sheet_name='onshore') # Assuming 'Onshore' is the sheet name for wind data
    print("Onshore wind data loaded successfully.")
except Exception as e:
    print(f"Error loading onshore wind data: {e}")
    print("Please ensure the Excel file path and sheet name ('Onshore') are correct.")
    sys.exit() # Exit if data can't be loaded

# Convert capacity column to numeric for wind_df
if 'Expected capacity (MW)' in wind_df.columns:
    wind_df['Expected capacity (MW)'] = pd.to_numeric(wind_df['Expected capacity (MW)'], errors='coerce')
else:
    print("Error: 'Expected capacity (MW)' column not found in wind data.")
    print(f"Available columns: {wind_df.columns.tolist()}")
    sys.exit()

# --- Data Validation and Info for Wind Data ---
print("\n--- Onshore Wind Data Info ---")
print("Available columns:", wind_df.columns.tolist())
print(f"Total rows in DataFrame: {len(wind_df)}")

# Check for essential columns
required_cols = ['Longitude', 'Latitude', 'Expected capacity (MW)', 'Project']
missing_cols = [col for col in required_cols if col not in wind_df.columns]
if missing_cols:
    print(f"Error: Missing required columns in wind data: {', '.join(missing_cols)}")
    sys.exit()

print(f"Rows with valid Longitude: {wind_df['Longitude'].notna().sum()}")
print(f"Rows with valid Latitude: {wind_df['Latitude'].notna().sum()}")
print(f"Rows with valid Expected capacity (MW) after conversion: {wind_df['Expected capacity (MW)'].notna().sum()}")

print("\nCoordinate ranges (Original Wind Data):")
if wind_df['Longitude'].notna().any() and wind_df['Latitude'].notna().any():
    print(f"Original Longitude column range: {wind_df['Longitude'].min()} to {wind_df['Longitude'].max()}")
    print(f"Original Latitude column range: {wind_df['Latitude'].min()} to {wind_df['Latitude'].max()}")
else:
    print("Longitude or Latitude data is missing or all NaN.")

# Read the extracted GADM GeoJSON file for Vietnam boundaries
vietnam = gpd.read_file(json_path_gadm)
print("\nVietnam GeoJSON for boundaries loaded.")

# Create the visualization
fig, ax = plt.subplots(1, 1, figsize=(15, 12))

# Set up the base map with province boundaries
vietnam.plot(ax=ax, color='lightgray', alpha=0.3)
vietnam.boundary.plot(ax=ax, linewidth=0.5, color='black')

# Set the axis limits to Vietnam's boundaries (with a small buffer)
vietnam_bounds = vietnam.total_bounds
buffer = 0.5  # degrees
ax.set_xlim([vietnam_bounds[0] - buffer, vietnam_bounds[2] + buffer])
ax.set_ylim([vietnam_bounds[1] - buffer, vietnam_bounds[3] + buffer])

# Convert wind_df to GeoDataFrame
# IMPORTANT: Assuming Longitude is X and Latitude is Y. If your data has them swapped, adjust here.
# Based on previous debugging, it's likely Latitude is X and Longitude is Y for your data.
gdf_projects_wind = gpd.GeoDataFrame(
    wind_df,
    geometry=gpd.points_from_xy(wind_df.Latitude, wind_df.Longitude), # Swapped based on previous solar data
    crs="EPSG:4326"
)

print(f"\nWind GeoDataFrame created with {len(gdf_projects_wind)} points")
print(f"Valid geometries in Wind GeoDataFrame: {gdf_projects_wind.geometry.notna().sum()}")
if gdf_projects_wind.geometry.notna().any():
    print(f"Wind Geometry bounds: {gdf_projects_wind.total_bounds}")
else:
    print("No valid geometries in Wind GeoDataFrame.")


# Check if CRS transformation is needed
print(f"Vietnam CRS: {vietnam.crs}")
print(f"Wind Projects CRS: {gdf_projects_wind.crs}")

if vietnam.crs != gdf_projects_wind.crs:
    print("Converting Wind Projects CRS to match Vietnam CRS...")
    gdf_projects_wind = gdf_projects_wind.to_crs(vietnam.crs)

# Create a colormap for capacity values (can be the same or different for wind)
colors_wind = ['#ADD8E6', '#87CEEB', '#00BFFF', '#1E90FF', '#0000FF', '#00008B'] # Example: Blueish colormap for wind
cmap_wind = LinearSegmentedColormap.from_list('wind_cmap', colors_wind)

# Normalize capacity for sizing and coloring
max_capacity_wind = wind_df['Expected capacity (MW)'].max()
min_capacity_wind = wind_df['Expected capacity (MW)'].min()
min_size, max_size = 30, 500 # Can adjust these values

# Create a custom normalization function for better visual representation
def scale_marker_size_wind(capacity):
    if pd.isna(capacity) or pd.isna(max_capacity_wind) or max_capacity_wind == 0:
        return min_size
    capacity = max(0, capacity) # Ensure capacity is non-negative
    return min_size + np.sqrt(capacity / max_capacity_wind) * (max_size - min_size)

# Filter for valid projects to plot (valid capacity and geometry)
valid_projects_wind = gdf_projects_wind.dropna(subset=['Expected capacity (MW)', 'geometry'])
print(f"\nValid onshore wind projects for plotting: {len(valid_projects_wind)} of {len(gdf_projects_wind)}")

if len(valid_projects_wind) == 0:
    print("WARNING: No valid onshore wind projects to plot!")
else:
    print("Sample of valid onshore wind projects:")
    print(valid_projects_wind[['Project', 'Expected capacity (MW)', 'geometry']].head())

    # Plot each project as a circle
    scatter_wind = ax.scatter(
        valid_projects_wind.geometry.x,
        valid_projects_wind.geometry.y,
        s=valid_projects_wind['Expected capacity (MW)'].apply(lambda x: max(10, scale_marker_size_wind(x))),
        c=valid_projects_wind['Expected capacity (MW)'],
        cmap=cmap_wind,
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5,
        zorder=10
    )
    print(f"Scatter plot for onshore wind created with {len(valid_projects_wind)} points")

    # # Add labels for larger projects
    # for idx, project in valid_projects_wind.nlargest(10, 'Expected capacity (MW)').iterrows():
    #     capacity = project['Expected capacity (MW)']
    #     if project.geometry.is_valid and pd.notna(capacity):
    #         plt.annotate(
    #             f"{project['Project']}\n({capacity:.0f} MW)",
    #             (project.geometry.x, project.geometry.y),
    #             fontsize=8,
    #             ha='center',
    #             va='bottom',
    #             xytext=(0, 5),
    #             textcoords='offset points',
    #             bbox=dict(boxstyle="round,pad=0.2", fc='white', ec='gray', alpha=0.7)
    #         )

    # Add a colorbar
    cbar_wind = plt.colorbar(scatter_wind, ax=ax, shrink=0.7)
    cbar_wind.set_label('Expected Capacity (MW) - Onshore Wind', fontsize=10)

    # Add a legend for marker sizes
    if pd.notna(max_capacity_wind) and max_capacity_wind > 0:
        capacity_examples_wind = [max_capacity_wind, max_capacity_wind/2, max_capacity_wind/5]
        legend_handles = []
        for capacity in capacity_examples_wind:
            if pd.notna(capacity) and capacity > 0: # Ensure capacity example is valid
                legend_handles.append(ax.scatter([], [], s=scale_marker_size_wind(capacity), color='gray', edgecolor='black',
                                   label=f"{capacity:.0f} MW"))
        if legend_handles:
             ax.legend(handles=legend_handles, title="Project Capacity", loc="lower right", frameon=True, fontsize=9)
        else:
            print("Could not generate capacity examples for legend.")

    else:
        print("Skipping capacity legend for wind due to invalid max_capacity_wind.")

# Add a basemap for context
try:
    ctx.add_basemap(
        ax,
        crs=vietnam.crs.to_string(),
        source=ctx.providers.CartoDB.Positron, # You can try other providers too
        zoom=7, # Adjust zoom level as needed
        attribution_size=8
    )
except Exception as e:
    print(f"Could not add basemap: {e}")

# Add title and remove axes
ax.set_title('Onshore Wind Power Projects in Vietnam (PDP8)', fontsize=15)
ax.set_axis_off()

# # Add summary statistics
# total_capacity_wind = wind_df['Expected capacity (MW)'].sum() # Sum of all valid numeric capacities
# num_plotted_projects_wind = len(valid_projects_wind)
# textstr_wind = (f"Total Onshore Wind Capacity: {total_capacity_wind:.0f} MW\n"
#                 f"Number of Plotted Projects: {num_plotted_projects_wind}")
# props_wind = dict(boxstyle='round', facecolor='white', alpha=0.8)
# plt.text(0.05, 0.05, textstr_wind, transform=ax.transAxes, fontsize=10,
#          verticalalignment='bottom', bbox=props_wind)

# Save and display the map
plt.tight_layout()
plt.savefig('vietnam_onshore_wind_projects_map.png', dpi=300)
print("\nMap saved as 'vietnam_onshore_wind_projects_map.png'")
plt.show()

# Clean up the temporary directory
try:
    import shutil
    shutil.rmtree(temp_dir)
    print(f"Temporary directory {temp_dir} removed.")
except Exception as e:
    print(f"Error removing temporary directory {temp_dir}: {e}")
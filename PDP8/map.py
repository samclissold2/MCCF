import pandas as pd
import folium
import branca.colormap as cm
from folium.plugins import MarkerCluster, Fullscreen, MiniMap, MeasureControl, HeatMap
from pathlib import Path
import webbrowser
import os
import requests
import json
import re
import numpy as np
import argparse
from config import (
    INFRASTRUCTURE_DATA,
    PDP8_PROJECT_DATA,
    PROJECTS_MAP,
    SUBSTATION_MAP,
    TRANSMISSION_MAP,
    INTEGRATED_MAP,
    DATA_DIR,
    RESULTS_DIR
)
from gpkg_utils import get_substations, get_power_lines, get_power_towers
from sklearn.cluster import DBSCAN
import geopandas as gpd
import joblib

def read_infrastructure_data():
    """
    Reads the infrastructure_data.xlsx file and returns a DataFrame.
    """
    if not INFRASTRUCTURE_DATA.exists():
        raise FileNotFoundError(f"Could not find {INFRASTRUCTURE_DATA}")
    df = pd.read_excel(INFRASTRUCTURE_DATA)
    return df

def read_and_clean_power_data():
    sheet_names = [
        "solar",
        "onshore",
        "LNG-fired gas",
        'cogeneration',
        'domestic gas-fired',
        'hydro',
        'pumped-storage',
        'nuclear',
        'biomass',
        'waste-to-energy',
        'flexible'
    ]
    if not PDP8_PROJECT_DATA.exists():
        raise FileNotFoundError(f"Could not find {PDP8_PROJECT_DATA}")
    df_dict = pd.read_excel(PDP8_PROJECT_DATA, sheet_name=sheet_names, engine="openpyxl")
    lat_col  = "longitude"
    lon_col  = "latitude"
    mw_col   = "expected capacity mw"
    name_col = "project"
    all_dfs = []
    for tech, df in df_dict.items():
        df.columns = (
            df.columns.str.strip().str.lower()
              .str.replace(r"[^\w\s]", "", regex=True)
              .str.replace(r"\s+", " ", regex=True)
        )
        df = df[df[name_col].astype(str).str.strip().astype(bool)]
        df["tech"] = tech.title()
        phases = df["operational phase"].dropna().unique()
        for phase in phases:
            col_name = f"phase_{str(phase).replace('-', '_')}"
            df[col_name] = df["operational phase"] == phase
        df["operational phase original"] = df["operational phase"]
        def assign_period(phase):
            if pd.isna(phase):
                return np.nan
            s = str(phase).strip()
            m = re.fullmatch(r"(\d{4})", s)
            if m:
                year = int(m.group(1))
                if 2025 <= year <= 2030:
                    return "2025-2030"
                elif 2031 <= year <= 2035:
                    return "2031-2035"
                else:
                    return np.nan
            m = re.fullmatch(r"(\d{4})\s*-\s*(\d{4})", s)
            if m:
                max_year = int(m.group(2))
                if 2025 <= max_year <= 2030:
                    return "2025-2030"
                elif 2031 <= max_year <= 2035:
                    return "2031-2035"
                else:
                    return np.nan
            if s in ["2025-2030", "2031-2035"]:
                return s
            m = re.findall(r"\d{4}", s)
            if m:
                max_year = max(map(int, m))
                if 2025 <= max_year <= 2030:
                    return "2025-2030"
                elif 2031 <= max_year <= 2035:
                    return "2031-2035"
            return np.nan
        df["period"] = df["operational phase"].apply(assign_period)
        df = df.dropna(subset=["period"])
        all_dfs.append(df)
    df = pd.concat(all_dfs, ignore_index=True)
    df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["mw"]  = (
        df[mw_col]
          .astype(str)
          .str.replace(",", ".")
          .pipe(pd.to_numeric, errors="coerce")
    )
    df = df.dropna(subset=["lat", "lon", "mw"])
    return df, name_col

#for test push

def read_solar_irradiance_points():
    """
    Reads the extracted solar irradiance CSV and returns a DataFrame.
    """
    csv_path = DATA_DIR / "extracted_data" / "solar_irradiance_points.csv"
    if not csv_path.exists():
        print(f"Solar irradiance CSV not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    return df

def create_folium_map(df):
    m = folium.Map(
        location=[df.lat.mean(), df.lon.mean()],
        zoom_start=6,
        tiles="CartoDB Positron",
        attr="© OpenStreetMap, © CartoDB",
    )
    for Plugin in (Fullscreen, MiniMap, MeasureControl):
        Plugin().add_to(m)
    legend_html = '''
    <div id="tech-legend" style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 220px; z-index:9999; 
        background: white; border:2px solid grey; border-radius:6px; 
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3); 
        font-size:14px; padding: 10px;">
      <div style="cursor:pointer;font-weight:bold;">
        Technology Legend
      </div>
      <div id="legend-content" style="display:block; margin-top:8px;">
        <div><span style="background:#FFA500; width:16px; height:16px; display:inline-block; border-radius:50%;"></span> Solar</div>
        <div><span style="background:#003366; width:16px; height:16px; display:inline-block; border-radius:50%;"></span> Hydro</div>
        <div><span style="background:#87CEEB; width:16px; height:16px; display:inline-block; border-radius:50%;"></span> Onshore</div>
        <div><span style="background:#D3D3D3; width:16px; height:16px; display:inline-block; border-radius:50%;"></span> LNG-Fired Gas</div>
        <div><span style="background:#333333; width:16px; height:16px; display:inline-block; border-radius:50%;"></span> Domestic Gas-Fired</div>
        <div><span style="background:#4682B4; width:16px; height:16px; display:inline-block; border-radius:50%;"></span> Pumped-Storage</div>
        <div><span style="background:#800080; width:16px; height:16px; display:inline-block; border-radius:50%;"></span> Nuclear</div>
        <div><span style="background:#228B22; width:16px; height:16px; display:inline-block; border-radius:50%;"></span> Biomass</div>
        <div><span style="background:#8B6F22; width:16px; height:16px; display:inline-block; border-radius:50%;"></span> Waste-To-Energy</div>
        <div><span style="background:#000000; width:16px; height:16px; display:inline-block; border-radius:50%;"></span> Flexible</div>
      </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Overlay solar irradiance heatmap
    solar_df = read_solar_irradiance_points()
    if solar_df is not None and not solar_df.empty:
        heat_data = [
            [row['lat'], row['lon'], row['irradiance']] for _, row in solar_df.iterrows()
        ]
        # More transparent heatmap
        heatmap_layer = HeatMap(
            heat_data,
            name="Solar Irradiance Heatmap",
            min_opacity=0.005,
            max_opacity=0.015,
            radius=8,
            blur=12,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'},
            show=True  # Start with the layer ON
        )
        heatmap_fg = folium.FeatureGroup(name="Solar Irradiance Heatmap", show=True)
        heatmap_fg.add_child(heatmap_layer)
        m.add_child(heatmap_fg)

    return m

def add_project_layers(m, df, name_col):
    tech_colors = {
        'Solar': '#FF0000',            # red
        'Hydro': '#003366',            # dark blue
        'Onshore': '#87CEEB',          # light blue
        'LNG-Fired Gas': '#D3D3D3',    # light grey
        'Domestic Gas-Fired': '#333333', # dark grey
        'Pumped-Storage': '#4682B4',   # medium blue
        'Nuclear': '#800080',          # purple
        'Biomass': '#228B22',          # green
        'Waste-To-Energy': '#8B6F22',  # dirty green/brown
        'Flexible': '#000000',         # black
    }
    groups = {}
    # Only show Solar 2025-2030 and Solar 2031-2035 by default
    for tech in df.tech.unique():
        for period in df[df.tech == tech]["period"].unique():
            name = f"{tech} {period}"
            show = (tech == 'Solar' and period in ['2025-2030', '2031-2035'])
            fg = folium.FeatureGroup(name=name, show=show).add_to(m)
            groups[(tech, period)] = fg
    for _, row in df.iterrows():
        fg = groups[(row.tech, row["period"])]
        color = tech_colors.get(row.tech, '#888888')
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=max(4, (row.mw ** 0.5) * 0.5),
            color='#222',  # faint outline
            opacity=0.3,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=1,
            tooltip=f"{row[name_col]} ({row.tech}, {row['period']}) — {row.mw:.0f} MW"
        ).add_to(fg)
    # Add substations as a toggleable layer
    sdf = read_substation_data()
    sdf = sdf[sdf['substation_type'].notna() & (sdf['substation_type'].astype(str).str.strip() != '')]
    if not sdf.empty:
        sub_fg = folium.FeatureGroup(name="Substations", show=False).add_to(m)
        for _, row in sdf.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                icon=folium.DivIcon(html='<div style="font-size:10px; color:#000; font-weight:bold;">×</div>'),
                tooltip="Substation"
            ).add_to(sub_fg)
    folium.LayerControl(collapsed=False).add_to(m)
    
def save_and_open_map(m, output_file=None):
    if output_file is None:
        output_file = PROJECTS_MAP
    m.save(output_file)
    print(f"🗺  Plotted map → {output_file}")
    file_path = os.path.abspath(output_file)
    webbrowser.open(f"file://{file_path}")
    print(f"🌐 Opening map in browser...")

def read_transmission_data():
    """
    Reads infrastructure_data.xlsx and returns a DataFrame with rows where 'location' is not blank.
    Only keeps longitude and latitude columns and relevant info.
    """
    if not INFRASTRUCTURE_DATA.exists():
        raise FileNotFoundError(f"Could not find {INFRASTRUCTURE_DATA}")
    df = pd.read_excel(INFRASTRUCTURE_DATA)
    # Filter for rows where 'location' is not blank
    df = df[df['location'].astype(str).str.strip().astype(bool)]
    # Only keep rows where location is 'overhead' or 'underground'
    df = df[df['location'].str.lower().isin(['overhead', 'underground'])]
    # Only keep relevant columns
    keep_cols = ['location', 'longitude', 'latitude']
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols]
    return df

def create_transmission_map(df):
    """
    Builds a folium map for transmission lines (overhead/underground).
    """
    m = folium.Map(
        location=[df['latitude'].mean(), df['longitude'].mean()],
        zoom_start=6,
        tiles="CartoDB Positron",
        attr="© OpenStreetMap, © CartoDB",
    )
    color_map = {'overhead': 'red', 'underground': 'blue'}
    cluster = MarkerCluster().add_to(m)
    for _, row in df.iterrows():
        loc_type = row['location'].lower()
        color = color_map.get(loc_type, 'gray')
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=1,
            tooltip=f"{row['location'].capitalize()} Transmission Line"
        ).add_to(cluster)
    legend_html = '''
    <div id="trans-legend" style="
        position: fixed; 
        bottom: 50px; left: 50px; width: 180px; z-index:9999; 
        background: white; border:2px solid grey; border-radius:6px; 
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3); 
        font-size:14px; padding: 10px;">
      <div style="cursor:pointer;font-weight:bold;">
        Transmission Legend
      </div>
      <div id="legend-content" style="display:block; margin-top:8px;">
        <div><span style="background:red; width:16px; height:16px; display:inline-block; border-radius:50%;"></span> Overhead</div>
        <div><span style="background:blue; width:16px; height:16px; display:inline-block; border-radius:50%;"></span> Underground</div>
      </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

def read_substation_data():
    """
    Reads infrastructure_data.xlsx and returns a DataFrame with rows where 'substation_type' is not blank.
    Only keeps longitude and latitude columns and relevant info.
    """
    if not INFRASTRUCTURE_DATA.exists():
        raise FileNotFoundError(f"Could not find {INFRASTRUCTURE_DATA}")
    df = pd.read_excel(INFRASTRUCTURE_DATA)
    # Filter for rows where 'substation_type' is not blank
    df = df[df['substation_type'].astype(str).str.strip().astype(bool)]
    # Only keep relevant columns

    keep_cols = ['substation_type', 'max_voltage', 'longitude', 'latitude']
    keep_cols = [col for col in keep_cols if col in df.columns]
    df = df[keep_cols]
    return df

def create_substation_map():
    df = read_substation_data()
    # Remove rows where substation_type is NaN or empty after stripping
    df = df[df['substation_type'].notna() & (df['substation_type'].astype(str).str.strip() != '')]
    m = folium.Map(
        location=[df['latitude'].mean(), df['longitude'].mean()],
        zoom_start=6,
        tiles="CartoDB Positron",
        attr="© OpenStreetMap, © CartoDB",
    )
    # Plot all substations as black dots, no clustering, no type distinction
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            icon=folium.DivIcon(html='<div style="font-size:10px; color:#000; font-weight:bold;">×</div>'),
            tooltip="Substation"
        ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    # Add a simple legend for substations
    legend_html = '<div id="substation-legend" style="position: fixed; bottom: 50px; left: 50px; width: 140px; z-index:9999; background: white; border:2px solid grey; border-radius:6px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3); font-size:12px; padding: 8px;">'
    legend_html += '<div style="cursor:pointer;font-weight:bold;">Substation Legend</div>'
    legend_html += '<div id="legend-content" style="display:block; margin-top:6px;">'
    legend_html += '<div><span style="background:#000; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Substation</div>'
    legend_html += '</div></div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

def cache_polylines(gdf, cache_file='powerline_polylines.geojson', eps=0.0025, min_samples=3, force_recompute=False):
    """
    Cluster power lines by voltage category, then order with greedy path within each voltage group.
    Returns a list of polylines (each is a list of (lat, lon) tuples).
    """
    if os.path.exists(cache_file) and not force_recompute:
        print(f"Loading polylines from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            geojson = json.load(f)
        return geojson['features']
    
    print("Computing DBSCAN clusters and greedy paths for each voltage category...")
    features = []
    
    # Group power lines by voltage category
    voltage_groups = gdf.groupby('voltage_cat')
    
    for voltage_cat, group in voltage_groups:
        if len(group) < min_samples:
            continue
            
        # Get coordinates of power line endpoints
        coords = []
        for geom in group.geometry:
            if geom.geom_type == 'LineString':
                coords.extend(geom.coords)  # coords are already (x,y) tuples
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    coords.extend(line.coords)  # coords are already (x,y) tuples
        
        if not coords:
            continue
            
        coords = np.array(coords)
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        
        # Process each cluster within this voltage category
        for cluster_id in set(db.labels_):
            if cluster_id == -1:
                continue  # noise
                
            cluster_points = coords[db.labels_ == cluster_id]
            # Greedy path ordering
            path = [cluster_points[0]]
            used = set([0])
            
            for _ in range(1, len(cluster_points)):
                last = path[-1]
                dists = np.linalg.norm(cluster_points - last, axis=1)
                dists[list(used)] = np.inf
                next_idx = np.argmin(dists)
                path.append(cluster_points[next_idx])
                used.add(next_idx)
                
            # Save as GeoJSON LineString with voltage category
            line_coords = [[float(y), float(x)] for x, y in path]
            features.append({
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": line_coords},
                "properties": {
                    "cluster": int(cluster_id),
                    "voltage": voltage_cat
                }
            })
    
    geojson = {"type": "FeatureCollection", "features": features}
    with open(cache_file, 'w') as f:
        json.dump(geojson, f)
    print(f"Saved polylines to {cache_file}")
    return features

def create_powerline_map(force_recompute=False):
    gdf = get_power_lines()
    towers_gdf = get_power_towers()
    # Categorize by max_voltage for lines
    def voltage_category(val):
        try:
            v = float(val)
            if v >= 500000:
                return '500kV'
            elif v >= 220000:
                return '220kV'
            elif v >= 115000:
                return '115kV'
            elif v >= 110000:
                return '110kV'
            elif v >= 50000:
                return '50kV'
            elif v >= 33000:
                return '33kV'
            elif v >= 25000:
                return '25kV'
            elif v >= 22000:
                return '22kV'
            else:
                return '<22kV'
        except:
            return 'Unknown'

    if 'max_voltage' in gdf.columns:
        gdf['voltage_cat'] = gdf['max_voltage'].apply(voltage_category)
    else:
        gdf['voltage_cat'] = 'Unknown'
    
    voltage_colors = {
        '500kV': 'red',
        '220kV': 'orange',
        '115kV': 'purple',
        '110kV': 'blue',
        '50kV': 'green',
        '33kV': 'brown',
        '25kV': 'pink',
        '22kV': 'gray',
        '<22kV': 'black',
        'Unknown': 'black',
    }
    
    m = folium.Map(
        location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()],
        zoom_start=6,
        tiles="CartoDB Positron",
        attr="© OpenStreetMap, © CartoDB",
    )
    
    # Plot polylines from cache, colored by voltage category
    features = cache_polylines(gdf, cache_file='powerline_polylines.geojson', eps=0.0025, min_samples=3, force_recompute=force_recompute)
    
    for feat in features:
        coords = feat['geometry']['coordinates']
        voltage = feat['properties']['voltage']
        color = voltage_colors.get(voltage, 'black')
        folium.PolyLine(
            locations=[(lat, lon) for lat, lon in coords],
            color=color,
            weight=4,
            opacity=0.7,
            tooltip=f"{voltage} Line (Cluster {feat['properties']['cluster']})"
        ).add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    legend_html = '''
    <div id="powerline-legend" style="position: fixed; bottom: 50px; left: 50px; width: 400px; z-index:9999; background: white; border:2px solid grey; border-radius:6px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3); font-size:14px; padding: 10px;">
      <div style="cursor:pointer;font-weight:bold;">Transmission Line Legend</div>
      <div id="legend-content" style="display:block; margin-top:8px;">
        <div><span style="display:inline-block; width:32px; height:4px; background:red; margin-right:4px;"></span> 500kV Lines</div>
        <div><span style="display:inline-block; width:32px; height:4px; background:orange; margin-right:4px;"></span> 220kV Lines</div>
        <div><span style="display:inline-block; width:32px; height:4px; background:purple; margin-right:4px;"></span> 115kV Lines</div>
        <div><span style="display:inline-block; width:32px; height:4px; background:blue; margin-right:4px;"></span> 110kV Lines</div>
        <div><span style="display:inline-block; width:32px; height:4px; background:green; margin-right:4px;"></span> 50kV Lines</div>
        <div><span style="display:inline-block; width:32px; height:4px; background:brown; margin-right:4px;"></span> 33kV Lines</div>
        <div><span style="display:inline-block; width:32px; height:4px; background:pink; margin-right:4px;"></span> 25kV Lines</div>
        <div><span style="display:inline-block; width:32px; height:4px; background:gray; margin-right:4px;"></span> 22kV Lines</div>
        <div><span style="display:inline-block; width:32px; height:4px; background:black; margin-right:4px;"></span> <22kV or Unknown Lines</div>
        <div style="margin-top:8px;">Lines are clustered by voltage category and spatial proximity.</div>
      </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

def create_integrated_map(force_recompute=False):
    """
    Creates a map that overlays power projects, substations, and power lines.
    """
    # Create base map
    m = folium.Map(
        location=[21.0, 105.8],  # Center of Vietnam
        zoom_start=6,
        tiles="CartoDB Positron",
        attr="© OpenStreetMap, © CartoDB",
    )
    
    # Add power lines
    gdf = get_power_lines()
    def voltage_category(val):
        try:
            v = float(val)
            if v >= 500000:
                return '500kV'
            elif v >= 220000:
                return '220kV'
            elif v >= 115000:
                return '115kV'
            elif v >= 110000:
                return '110kV'
            elif v >= 50000:
                return '50kV'
            elif v >= 33000:
                return '33kV'
            elif v >= 25000:
                return '25kV'
            elif v >= 22000:
                return '22kV'
            else:
                return '<22kV'
        except:
            return 'Unknown'
    if 'max_voltage' in gdf.columns:
        gdf['voltage_cat'] = gdf['max_voltage'].apply(voltage_category)
    else:
        gdf['voltage_cat'] = 'Unknown'
    features = cache_polylines(gdf, cache_file='powerline_polylines.geojson', eps=0.005, min_samples=4, force_recompute=force_recompute)
    voltage_colors = {
        '500kV': 'red',
        '220kV': 'orange',
        '115kV': 'purple',
        '110kV': 'blue',
        '50kV': 'green',
        '33kV': 'brown',
        '25kV': 'pink',
        '22kV': 'gray',
        '<22kV': 'black',
        'Unknown': 'black',
    }
    
    for feat in features:
        coords = feat['geometry']['coordinates']
        voltage = feat['properties']['voltage']
        color = voltage_colors.get(voltage, 'black')
        folium.PolyLine(
            locations=[(lat, lon) for lat, lon in coords],
            color=color,
            weight=4,
            opacity=0.7,
            tooltip=f"{voltage} Line (Cluster {feat['properties']['cluster']})"
        ).add_to(m)
    
    # Add substations
    sdf = read_substation_data()
    sdf = sdf[sdf['substation_type'].notna() & (sdf['substation_type'].astype(str).str.strip() != '')]
    # 3. Bucket substations by max_voltage and assign colors
    def voltage_category(val):
        try:
            v = float(val)
            if v >= 500000:
                return '500kV'
            elif v >= 220000:
                return '220kV'
            elif v >= 115000:
                return '115kV'
            elif v >= 110000:
                return '110kV'
            elif v >= 50000:
                return '50kV'
            elif v >= 33000:
                return '33kV'
            elif v >= 25000:
                return '25kV'
            elif v >= 22000:
                return '22kV'
            else:
                return '<22kV'
        except:
            return 'Unknown'
    if 'max_voltage' in sdf.columns:
        sdf['voltage_cat'] = sdf['max_voltage'].apply(voltage_category)
    else:
        sdf['voltage_cat'] = 'Unknown'
    voltage_colors = {
        '500kV': 'red',
        '220kV': 'orange',
        '115kV': 'purple',
        '110kV': 'blue',
        '50kV': 'green',
        '33kV': 'brown',
        '25kV': 'pink',
        '22kV': 'gray',
        '<22kV': 'black',
        'Unknown': 'black',
    }
    if not sdf.empty:
        sub_fg = folium.FeatureGroup(name="Substations", show=False).add_to(m)
        for _, row in sdf.iterrows():
            color = voltage_colors.get(row['voltage_cat'], 'black')
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                icon=folium.DivIcon(html=f'<div style="font-size:10px; color:{color}; font-weight:bold;">×</div>'),
                tooltip=f"Substation ({row['voltage_cat']})"
            ).add_to(sub_fg)
    # Add substation voltage legend
    substation_legend = '''
    <div id="substation-legend" style="position: fixed; top: 180px; left: 50px; width: 180px; z-index:9999; background: white; border:2px solid grey; border-radius:6px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3); font-size:12px; padding: 8px;">
      <div style="cursor:pointer;font-weight:bold;">Substation Voltage Legend</div>
      <div id="legend-content" style="display:block; margin-top:6px;">
        <div><span style="display:inline-block; width:16px; height:16px; background:red; border-radius:50%; margin-right:4px;"></span> 500kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:orange; border-radius:50%; margin-right:4px;"></span> 220kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:purple; border-radius:50%; margin-right:4px;"></span> 115kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:blue; border-radius:50%; margin-right:4px;"></span> 110kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:green; border-radius:50%; margin-right:4px;"></span> 50kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:brown; border-radius:50%; margin-right:4px;"></span> 33kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:pink; border-radius:50%; margin-right:4px;"></span> 25kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:gray; border-radius:50%; margin-right:4px;"></span> 22kV</div>
        <div><span style="display:inline-block; width:16px; height:16px; background:black; border-radius:50%; margin-right:4px;"></span> <22kV or Unknown</div>
      </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(substation_legend))
    # Add solar irradiance heatmap to integrated map, toggled off by default
    solar_df = read_solar_irradiance_points()
    if solar_df is not None and not solar_df.empty:
        heat_data = [
            [row['lat'], row['lon'], row['irradiance']] for _, row in solar_df.iterrows()
        ]
        heatmap_layer = HeatMap(
            heat_data,
            name="Solar Irradiance Heatmap",
            min_opacity=0.005,
            max_opacity=0.015,
            radius=8,
            blur=12,
            gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'},
            show=False
        )
        heatmap_fg = folium.FeatureGroup(name="Solar Irradiance Heatmap", show=False)
        heatmap_fg.add_child(heatmap_layer)
        m.add_child(heatmap_fg)
    
    # Add power projects
    df, name_col = read_and_clean_power_data()
    tech_colors = {
        'Solar': '#FF0000',            # red
        'Hydro': '#003366',            # dark blue
        'Onshore': '#87CEEB',          # light blue
        'LNG-Fired Gas': '#D3D3D3',    # light grey
        'Domestic Gas-Fired': '#333333', # dark grey
        'Pumped-Storage': '#4682B4',   # medium blue
        'Nuclear': '#800080',          # purple
        'Biomass': '#228B22',          # green
        'Waste-To-Energy': '#8B6F22',  # dirty green/brown
        'Flexible': '#000000',         # black
    }
    
    for tech in df.tech.unique():
        for period in df[df.tech == tech]["period"].unique():
            name = f"{tech} {period}"
            show = (tech == 'Solar' and period in ['2025-2030', '2031-2035'])
            fg = folium.FeatureGroup(name=name, show=show).add_to(m)
            tech_df = df[(df.tech == tech) & (df.period == period)]
            color = tech_colors.get(tech, '#888888')
            for _, row in tech_df.iterrows():
                folium.CircleMarker(
                    location=[row.lat, row.lon],
                    radius=max(4, (row.mw ** 0.5) * 0.5),
                    color='#222',  # faint outline
                    opacity=0.3,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.7,
                    weight=1,
                    tooltip=f"{row[name_col]} ({row.tech}, {row['period']}) — {row.mw:.0f} MW"
                ).add_to(fg)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add legends
    # Power line legend
    powerline_legend = '''
    <div id="powerline-legend" style="position: fixed; top: 50px; left: 50px; width: 180px; z-index:9999; background: white; border:2px solid grey; border-radius:6px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3); font-size:12px; padding: 8px;">
      <div style="cursor:pointer;font-weight:bold;">Transmission Line Legend</div>
      <div id="legend-content" style="display:block; margin-top:6px;">
        <div><span style="display:inline-block; width:24px; height:4px; background:red; margin-right:4px;"></span> 500kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:orange; margin-right:4px;"></span> 220kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:purple; margin-right:4px;"></span> 115kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:blue; margin-right:4px;"></span> 110kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:green; margin-right:4px;"></span> 50kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:brown; margin-right:4px;"></span> 33kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:pink; margin-right:4px;"></span> 25kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:gray; margin-right:4px;"></span> 22kV Lines</div>
        <div><span style="display:inline-block; width:24px; height:4px; background:black; margin-right:4px;"></span> <22kV or Unknown Lines</div>
      </div>
    </div>
    '''
    
    # Power project legend
    project_legend = '''
    <div id="project-legend" style="position: fixed; top: 320px; left: 50px; width: 180px; z-index:9999; background: white; border:2px solid grey; border-radius:6px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3); font-size:12px; padding: 8px;">
      <div style="cursor:pointer;font-weight:bold;">Power Project Legend</div>
      <div id="legend-content" style="display:block; margin-top:6px;">
        <div><span style="background:#FF0000; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Solar</div>
        <div><span style="background:#003366; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Hydro</div>
        <div><span style="background:#87CEEB; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Onshore</div>
        <div><span style="background:#D3D3D3; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> LNG-Fired Gas</div>
        <div><span style="background:#333333; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Domestic Gas-Fired</div>
        <div><span style="background:#4682B4; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Pumped-Storage</div>
        <div><span style="background:#800080; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Nuclear</div>
        <div><span style="background:#228B22; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Biomass</div>
        <div><span style="background:#8B6F22; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Waste-To-Energy</div>
        <div><span style="background:#000000; width:12px; height:12px; display:inline-block; border-radius:50%;"></span> Flexible</div>
      </div>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(powerline_legend))
    m.get_root().html.add_child(folium.Element(project_legend))
    
    return m

def read_existing_generators():
    """
    Extracts rows from infrastructure_data.xlsx where 'source' is not null/nan.
    Converts 'output' from kW to MW. If 'output' is missing but 'source' is present, fills with the average for that source group.
    Returns a DataFrame with columns: source, latitude, longitude, output_mw
    """
    if not INFRASTRUCTURE_DATA.exists():
        raise FileNotFoundError(f"Could not find {INFRASTRUCTURE_DATA}")
    df = pd.read_excel(INFRASTRUCTURE_DATA)
    # Only keep rows with a non-null, non-empty 'source'
    df = df[df['source'].notna() & (df['source'].astype(str).str.strip() != '')]
    # Only keep rows with lat/lon
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        raise ValueError("Missing latitude/longitude columns in infrastructure_data.xlsx")
    df = df[df['latitude'].notna() & df['longitude'].notna()]
    # Convert output to MW
    df['output_mw'] = pd.to_numeric(df['output'], errors='coerce') / 1000
    # Fill missing output_mw with group mean
    df['output_mw'] = df.groupby('source')['output_mw'].transform(lambda x: x.fillna(x.mean()))
    # If still missing (all NaN in group), fill with overall mean
    df['output_mw'] = df['output_mw'].fillna(df['output_mw'].mean())
    return df[['source', 'latitude', 'longitude', 'output_mw']]

def create_existing_generator_map():
    df = read_existing_generators()
    # Assign colors to each unique source
    unique_sources = df['source'].unique()
    color_palette = [
        '#FF0000', '#003366', '#87CEEB', '#D3D3D3', '#333333',
        '#4682B4', '#800080', '#228B22', '#8B6F22', '#000000',
        '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
        '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5',
        '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f',
        '#e5c494', '#b3b3b3'
    ]
    source_colors = {src: color_palette[i % len(color_palette)] for i, src in enumerate(unique_sources)}
    m = folium.Map(
        location=[df['latitude'].mean(), df['longitude'].mean()],
        zoom_start=6,
        tiles="CartoDB Positron",
        attr="© OpenStreetMap, © CartoDB",
    )
    for src in unique_sources:
        fg = folium.FeatureGroup(name=src).add_to(m)
        color = source_colors[src]
        sub_df = df[df['source'] == src]
        for _, row in sub_df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=max(4, (row['output_mw'] ** 0.5) * 0.5),
                color='#222',  # faint outline
                opacity=0.3,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                weight=1,
                tooltip=f"{src} — {row['output_mw']:.2f} MW"
            ).add_to(fg)
    folium.LayerControl(collapsed=False).add_to(m)
    # Add a legend for sources
    legend_html = '<div id="generator-legend" style="position: fixed; bottom: 50px; left: 50px; width: 260px; z-index:9999; background: white; border:2px solid grey; border-radius:6px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3); font-size:14px; padding: 10px;"><div style="cursor:pointer;font-weight:bold;">Existing Generator Source Legend</div><div id="legend-content" style="display:block; margin-top:8px;">'
    for src in unique_sources:
        legend_html += f'<div><span style="background:{source_colors[src]}; width:16px; height:16px; display:inline-block; border-radius:50%;"></span> {src}</div>'
    legend_html += '</div></div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    return m

def main():
    parser = argparse.ArgumentParser(description="Vietnam Power Maps")
    parser.add_argument('--map', choices=[
        'power', 'substation', 'transmission', 'integrated', 'existing_generator', 'all'
    ], default='power',
    help="Which map to generate: 'power', 'substation', 'transmission', 'integrated', 'existing_generator', or 'all'")
    parser.add_argument('--force-recompute', action='store_true', help='Force recompute the map(s) even if cache exists')
    args = parser.parse_args()

    if args.map in ['power', 'all']:
        df, name_col = read_and_clean_power_data()
        print(f"Loaded power data: {len(df)} rows")
        if df.empty:
            print("ERROR: Power data is empty. Map will not be generated.")
            return
        m = create_folium_map(df)
        add_project_layers(m, df, name_col)
        save_and_open_map(m, output_file=PROJECTS_MAP)
    if args.map in ['substation', 'all']:
        sdf = read_substation_data()
        if sdf.empty:
            print("No substation data found with non-blank 'substation_type'.")
        else:
            sm = create_substation_map()
            save_and_open_map(sm, output_file=SUBSTATION_MAP)
    if args.map in ['transmission', 'all']:
        force_recompute = args.force_recompute
        if not force_recompute:
            user_input = input("Do you want to force recompute the transmission clustering? (y/n): ").strip().lower()
            if user_input == 'y':
                force_recompute = True
        tm = create_powerline_map(force_recompute=force_recompute)
        save_and_open_map(tm, output_file=TRANSMISSION_MAP)
    if args.map in ['integrated', 'all']:
        force_recompute = args.force_recompute
        if not force_recompute:
            user_input = input("Do you want to force recompute the transmission clustering? (y/n): ").strip().lower()
            if user_input == 'y':
                force_recompute = True
        im = create_integrated_map(force_recompute=force_recompute)
        save_and_open_map(im, output_file=INTEGRATED_MAP)
    if args.map in ['existing_generator', 'all']:
        egm = create_existing_generator_map()
        save_and_open_map(egm, output_file=RESULTS_DIR / "vn_existing_generators_map.html")

if __name__ == "__main__":
    main()
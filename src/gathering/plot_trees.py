import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from typing import List, Tuple
from pathlib import Path
from datetime import date
from ghent_polygon import get_ghent_convex_hull, get_ghent_outers
from gather_trees import get_csv as get_trees

OUTPUT_FILE = Path(__file__).parent.absolute() / f"plot_trees.result.png"

# Load CSV
df = get_trees()

# Extract coordinates
df[['lat', 'lon']] = df['geo_point_2d'].str.split(',', expand=True).astype(float)

# Create Datashader canvas
canvas = ds.Canvas(plot_width=800, plot_height=600,
                   x_range=(df['lon'].min(), df['lon'].max()),
                   y_range=(df['lat'].min(), df['lat'].max()))
agg = canvas.points(df, 'lon', 'lat')
img = tf.shade(agg, cmap=["lightgreen", "darkgreen"], how='log')

# Convert to PIL for plotting with matplotlib
img_pil = img.to_pil()

# Plot with matplotlib
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(img_pil, extent=(df['lon'].min(), df['lon'].max(), df['lat'].min(), df['lat'].max()))

# Overlay convex hull polygon
convex_hull_coords = get_ghent_convex_hull()
convex_hull_lon, convex_hull_lat = zip(*convex_hull_coords)
ax.plot(convex_hull_lon, convex_hull_lat, color='red', linewidth=2, label='Convex Hull')


outers_coords = get_ghent_outers()
outers_lon, outers_lat = zip(*outers_coords)
ax.plot(outers_lon, outers_lat, color='blue', linewidth=2, label='Ghent Outers')


ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Ghent Trees with Convex Hull Overlay')
ax.legend()
plt.savefig(OUTPUT_FILE)
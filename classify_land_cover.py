import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

def main():
    print("=" * 60)
    print("      MULTISPECTRAL LAND COVER CLASSIFICATION PROGRAM       ")
    print("=" * 60)
    
    image_path = 'multispectral.tif'
    output_plot_path = 'land_cover_classification.png'
    
    if not os.path.exists(image_path):
        print(f"Error: Could not find '{image_path}' in the current directory.")
        return
        

    with rasterio.open(image_path) as src:
        # Read all bands (shape: 4, height, width)
        data = src.read()
        meta = src.meta
        width = src.width
        height = src.height
        
    print(f"Image dimensions: {width} x {height} pixels")
    print(f"Number of bands: {data.shape[0]}")
    
    # ----------------------------------------------------
    # Extract bands according to the physical sensor layout of multispectral.tif
    # Note: Visually and statistically verified as BGRN array to ensure correct 
    # NDVI calculation and natural color RGB compositing.
    # ----------------------------------------------------
    blue  = data[0].astype(np.float32)  # Band 1 in TIFF (Physical Blue)
    green = data[1].astype(np.float32)  # Band 2 in TIFF (Physical Green)
    red   = data[2].astype(np.float32)  # Band 3 in TIFF (Physical Red)
    nir   = data[3].astype(np.float32)  # Band 4 in TIFF (Physical NIR)
    
    # Create mask for valid data (pixels where all bands > 0)
    # Background/borders are 0
    valid_mask = (blue > 0) & (green > 0) & (red > 0) & (nir > 0)
    num_valid_pixels = np.sum(valid_mask)
    total_pixels = width * height
    print(f"Valid pixels: {num_valid_pixels:,} / {total_pixels:,} ({num_valid_pixels/total_pixels:.2%})")
    
    # ----------------------------------------------------
    # 1. Compute NDVI
    # ----------------------------------------------------
    print("Computing Normalized Difference Vegetation Index (NDVI)...")
    # NDVI = (NIR - Red) / (NIR + Red)
    # Handle division by zero using numpy arrays
    denom = nir + red
    ndvi = np.zeros_like(denom)
    # Only compute NDVI for valid pixels where denominator is non-zero
    calc_mask = valid_mask & (denom > 0)
    ndvi[calc_mask] = (nir[calc_mask] - red[calc_mask]) / denom[calc_mask]
    
    # Set background pixels to NaN for visualization and masking
    ndvi_vis = ndvi.copy()
    ndvi_vis[~valid_mask] = np.nan
    
    # ----------------------------------------------------
    # 2. RGB Combined (Natural Color) Creation
    # ----------------------------------------------------
    print("Generating natural color RGB visualization...")
    def scale_band(band, mask):
        # Clip to 2nd and 98th percentiles of valid pixels for high contrast
        valid_vals = band[mask]
        if len(valid_vals) == 0:
            return np.zeros_like(band)
        p2, p98 = np.percentile(valid_vals, (2, 98))
        if p98 == p2:
            p98 = p2 + 1.0
        scaled = np.clip((band - p2) / (p98 - p2), 0.0, 1.0)
        scaled[~mask] = 0.0 # Keep background black in RGB
        return scaled

    r_scaled = scale_band(red, valid_mask)
    g_scaled = scale_band(green, valid_mask)
    b_scaled = scale_band(blue, valid_mask)
    rgb_image = np.dstack((r_scaled, g_scaled, b_scaled))
    
    # ----------------------------------------------------
    # 3. K-Means Land Cover Clustering (k=5)
    # ----------------------------------------------------
    print("Preparing data and executing K-Means clustering (k=5)...")
    # Stack bands for clustering: shape (4, height, width) -> (height, width, 4)
    stacked_bands = np.dstack((blue, green, red, nir))
    # Reshape to (num_pixels, 4)
    flat_bands = stacked_bands.reshape(-1, 4)
    # Flat mask
    flat_mask = valid_mask.reshape(-1)
    # Extract only valid pixels for clustering to avoid clustering the background
    clustering_data = flat_bands[flat_mask]
    
    # Scale features using StandardScaler (crucial for distance-based algorithms like K-Means)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    
    # Perform K-Means clustering on scaled features
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    
    # Reconstruct cluster labels image
    # Pre-populate with -1 (representing no-data)
    classified_flat = np.full(flat_mask.shape, -1, dtype=np.int8)
    
    # Sort cluster labels by mean NDVI so the class numbers are physically meaningful:
    # 0 = lowest vegetation (soil, urban, built-up)
    # 4 = highest vegetation density (dense forest)
    # Reconstruct original centers using inverse_transform for physical analysis
    original_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    # Calculate NDVI of the cluster centers in physical space: center_NDVI = (NIR - Red) / (NIR + Red)
    # index 3 is NIR (band 4), index 2 is Red (band 3)
    center_ndvis = (original_centers[:, 3] - original_centers[:, 2]) / (original_centers[:, 3] + original_centers[:, 2])
    sorted_cluster_indices = np.argsort(center_ndvis)
    
    # Create mapping from sklearn cluster label to sorted label
    label_mapping = {old_label: new_label for new_label, old_label in enumerate(sorted_cluster_indices)}
    
    # Apply mapping
    sorted_labels = np.array([label_mapping[l] for l in kmeans.labels_])
    classified_flat[flat_mask] = sorted_labels
    
    # Reshape back to 2D image coordinates
    classified_img = classified_flat.reshape(height, width)
    
    # ----------------------------------------------------
    # 4. Correlation Matrix
    # ----------------------------------------------------
    print("Calculating correlation matrix between bands...")
    # Pearson correlation matrix using valid pixels
    correlation_matrix = np.corrcoef(clustering_data.T)
    band_names = ["Blue (B1)", "Green (B2)", "Red (B3)", "NIR (B4)"]
    
    # Print cluster statistics
    print("\nLand Cover Cluster Statistics (Sorted by NDVI):")
    for i in range(5):
        orig_label = sorted_cluster_indices[i]
        center = original_centers[orig_label]
        pixel_count = np.sum(sorted_labels == i)
        pct = pixel_count / num_valid_pixels
        ndvi_val = center_ndvis[orig_label]
        print(f"  Class {i} (NDVI ~ {ndvi_val:.3f}): {pixel_count:,} pixels ({pct:.2%})")
        print(f"    Mean Reflectance -> Blue: {center[0]:.1f}, Green: {center[1]:.1f}, Red: {center[2]:.1f}, NIR: {center[3]:.1f}")
        
    # ----------------------------------------------------
    # 5. Visualizing the results in a 2x4 Subplot Grid
    # ----------------------------------------------------
    print("\nGenerating final visualization plots...")
    
    # Setup premium styling
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    
    fig, axes = plt.subplots(2, 4, figsize=(24, 12), dpi=100)
    fig.suptitle("Multispectral Land Cover Analysis & K-Means Classification", fontsize=22, fontweight='bold', y=0.98)
    
    # Helper to apply standard visualization settings for single-band images
    def show_band(ax, band, title, cmap='gray'):
        # Mask nodata values
        vis_band = band.copy()
        vis_band[~valid_mask] = np.nan
        # Use 2%-98% stretch for grayscale rendering contrast
        vmin, vmax = np.percentile(band[valid_mask], (2, 98))
        im = ax.imshow(vis_band, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=14, fontweight='semibold', pad=10)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    # Row 1, Col 1-4: Individual Bands
    print("  Plotting individual bands...")
    show_band(axes[0, 0], red, "Red Band (Band 3)")
    show_band(axes[0, 1], green, "Green Band (Band 2)")
    show_band(axes[0, 2], blue, "Blue Band (Band 1)")
    show_band(axes[0, 3], nir, "NIR Band (Band 4)")
    
    # Row 2, Col 1: RGB Combined
    print("  Plotting RGB combined image...")
    axes[1, 0].imshow(rgb_image)
    axes[1, 0].set_title("RGB Combined (Natural Color)", fontsize=14, fontweight='semibold', pad=10)
    axes[1, 0].axis('off')
    
    # Row 2, Col 2: NDVI
    print("  Plotting NDVI map...")
    im_ndvi = axes[1, 1].imshow(ndvi_vis, cmap='RdYlGn', vmin=-0.1, vmax=0.9)
    axes[1, 1].set_title("NDVI (Veg. Index)", fontsize=14, fontweight='semibold', pad=10)
    axes[1, 1].axis('off')
    fig.colorbar(im_ndvi, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Row 2, Col 3: K-Means Clustering
    print("  Plotting K-Means classification map...")
    # Define a custom color palette for the 5 classes:
    # 0: Barren/Soil (Tan/Sandy)
    # 1: Sparse Veg/Grass (Yellow-Green)
    # 2: Moderate Veg (Light Green)
    # 3: Dense Veg (Green)
    # 4: Forest/Very Dense Veg (Dark Green)
    class_colors = ['#d2b48c', '#ccebc5', '#78c679', '#238443', '#004529']
    cmap_kmeans = ListedColormap(class_colors)
    
    # Mask out background (set to -1) in visualization. 
    # Matplotlib's colormap doesn't map -1, but we can set it to transparent or bad
    class_vis = classified_img.astype(float)
    class_vis[~valid_mask] = np.nan
    
    # Plot using our custom colormap
    # Using vmin=-0.5, vmax=4.5 to align discrete bins perfectly
    im_class = axes[1, 2].imshow(class_vis, cmap=cmap_kmeans, vmin=-0.5, vmax=4.5)
    axes[1, 2].set_title("K-Means (Euclidean, k=5)", fontsize=14, fontweight='semibold', pad=10)
    axes[1, 2].axis('off')
    
    # Add a custom categorical legend
    class_names = [
        "Class 0: Barren/Built-up",
        "Class 1: Sparse Vegetation",
        "Class 2: Moderate Veg / Crops",
        "Class 3: Dense Vegetation",
        "Class 4: Very Dense Forest"
    ]
    patches = [mpatches.Patch(color=class_colors[i], label=class_names[i]) for i in range(5)]
    axes[1, 2].legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.2), 
                      ncol=1, fontsize=9, frameon=True, facecolor='#f8f8f8', edgecolor='#dddddd')
    
    # Row 2, Col 4: Correlation Matrix Heatmap
    print("  Plotting band correlation matrix...")
    sns.heatmap(correlation_matrix, annot=True, fmt=".4f", cmap="coolwarm", vmin=-1.0, vmax=1.0,
                xticklabels=band_names, yticklabels=band_names, ax=axes[1, 4-1], cbar=True,
                square=True, annot_kws={"size": 11, "weight": "bold"})
    axes[1, 3].set_title("Band Correlation Matrix", fontsize=14, fontweight='semibold', pad=10)
    # Format labels to rotate nicely
    axes[1, 3].set_xticklabels(band_names, rotation=25, ha="right", fontsize=10)
    axes[1, 3].set_yticklabels(band_names, rotation=0, fontsize=10)
    
    # Clean layout and save
    plt.tight_layout()
    # Adjust top spacing to avoid title collision
    plt.subplots_adjust(top=0.90, bottom=0.08)
    
    print(f"Saving final plot to: {output_plot_path}...")
    plt.savefig(output_plot_path, dpi=150, bbox_inches='tight')
    
    print("Displaying plots on screen (blocking until window is closed)...")
    plt.show()
    print("Program completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

import sys
import os
import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from PyQt6.QtWidgets import QInputDialog, QMessageBox

FORCE_PROPERTY_ARRAY = {
    'fx': {'title': 'Force-Displacement Plot (Fx)', 'ylim': (-1, 1), 'ylabel': 'Force (N)', 'smooth_elements': 10, 'negate': False},
    'fy': {'title': 'Force-Displacement Plot (Fy)', 'ylim': (-1, 1), 'ylabel': 'Force (N)', 'smooth_elements': 10, 'negate': False},
    'fz': {'title': 'Force-Displacement Plot (Fz)', 'ylim': (-2.5, 6), 'ylabel': 'Force (N)', 'smooth_elements': 10, 'negate': False},
    'tx': {'title': 'Torque-Displacement Plot (Tx)', 'ylim': (-2, 14), 'ylabel': 'Torque (Nmm)', 'smooth_elements': 25, 'negate': False},
    'ty': {'title': 'Torque-Displacement Plot (Ty)', 'ylim': (-10, 50), 'ylabel': 'Torque (Nmm)', 'smooth_elements': 25, 'negate': False},
    'tz': {'title': 'Torque-Displacement Plot (Tz)', 'ylim': (-2, 14), 'ylabel': 'Torque (Nmm)', 'smooth_elements': 25, 'negate': False}
}

def get_folder_listing(folder_path, include_needle_number_bool=True):
    """Retrieve a list of unique sample names from CSV files in the specified directory."""
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    num_elements = 4 if include_needle_number_bool else 3
    sample_names = {"_".join(os.path.basename(file).split("_")[:num_elements]) for file in all_files}
    return sorted(sample_names)

def separate_needles_dialog():
    """Prompt user to select whether to separate needle samples."""
    reply = QMessageBox.question(None, "Separate Needles", "Do you want to handle needle samples separately?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    return reply == QMessageBox.StandardButton.Yes

def get_sample_names_dialog(all_files, multiple_samples=True):
    """Prompt user to select sample names for processing."""
    sample_names = []
    while all_files:
        sample_name, ok = QInputDialog.getItem(None, "Input Dialog", "Select a sample name:", all_files, 0, False)
        if not ok:
            break
        if not multiple_samples:
            return sample_name
        sample_names.append(sample_name)
        all_files.remove(sample_name)
        reply = QMessageBox.question(None, "Message", "Do you want to add another sample name?", QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            break
    return sample_names

def get_force_name_dialog():
    """Prompt user to select a force component for analysis."""
    force_name, ok = QInputDialog.getItem(None, "Input Dialog", "Select force component:", list(FORCE_PROPERTY_ARRAY.keys()), 2, False)
    if not ok:
        raise ValueError("No force entered.")
    return force_name

def calculate_metrics_dialog():
    """Prompt user to select whether to calculate metrics."""
    reply = QMessageBox.question(None, "Calculate Metrics", "Do you want to calculate the metrics?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    return reply == QMessageBox.StandardButton.Yes

def save_figure_dialog():
    """Prompt user to select whether to save the figure."""
    reply = QMessageBox.question(None, "Save Figure", "Do you want to save the figure?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    return reply == QMessageBox.StandardButton.Yes

def load_and_process_data(files, force_name, force_properties):
    """Load and process displacement and force data from files."""
    displacements, forces = [], []
    min_length = np.inf

    for file_path in files:
        data = pd.read_csv(file_path)
        displacement = data["pz"].values
        force_smooth = data[force_name].rolling(window=force_properties['smooth_elements']).mean().values
        if force_properties['negate']:
            force_smooth = -force_smooth
        min_length = min(min_length, len(displacement))
        displacements.append(displacement)
        forces.append(force_smooth)

    displacements = [d[:min_length] for d in displacements]
    forces = [f[:min_length] for f in forces]

    return np.array(displacements), np.array(forces)

def load_and_process_data_frames(files, force_name, force_properties):
    """Load and process displacement and force data from files."""
    pd_data_frames = pd.DataFrame()
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        parts = file_name.split('_')
        sample = parts[0]
        velocity = parts[1]
        angle = parts[2]
        needle = parts[3]
        sample_number = parts[4]

        data = pd.read_csv(file_path)
        displacement = data["pz"]
        force = data[force_name]

        if force_properties['negate']:
            force = -force

        force_smooth = force.rolling(window=force_properties['smooth_elements']).mean()

        pd_data_frame = pd.DataFrame({
            'Displacement': displacement,
            'Force': force_smooth,
            'Sample': f"{sample}\n({velocity} mm/s, {angle}\u00B0, Needle {needle}, Sample {sample_number})",
            'Sample Number': sample_number
        })

        pd_data_frames = pd.concat([pd_data_frames, pd_data_frame])
    
    return pd_data_frames

def find_insertion_peaks(force_smooth, max_displacement_index, distance, height, prominence):
    """Find peaks in the force data."""
    force_insertion = force_smooth[:max_displacement_index + 1]
    peak_indices, _ = find_peaks(force_insertion, distance=distance, height=height, prominence=prominence)
    return peak_indices

def find_retraction_maxima(force_smooth, max_displacement_index):
    """Find indices of maximum values for retraction."""
    force_retraction = force_smooth[max_displacement_index:]
    retraction_max_index = max_displacement_index + np.argmax(force_retraction)
    return retraction_max_index

def linear_regression(displacement, force_smooth, peak_indices, r_squared_threshold):
    """Perform linear regression and check R-squared value."""
    if len(peak_indices) > 0:
        first_peak_index = peak_indices[0]
        if first_peak_index > 200:
            for i in range(first_peak_index - 200, 0, -1):
                X = displacement[i:first_peak_index + 1].reshape(-1, 1)
                y = force_smooth[i:first_peak_index + 1]
                model = LinearRegression().fit(X, y)
                y_pred = model.predict(X)
                r_squared = r2_score(y, y_pred)

                if r_squared < r_squared_threshold:
                    return X, y_pred, model.coef_[0]
    return np.array([]), np.array([]), np.nan

def get_final_insertion_value(displacement, force_smooth):
    max_displacement_index = np.argmax(displacement)
    return max_displacement_index, force_smooth[max_displacement_index]

def calculate_force_metrics(displacement, force_smooth, distance_height_prominence = (250, 1.5, 0.5)):
    """Calculate metrics from the force and displacement data."""
    max_displacement_index, final_insertion_force = get_final_insertion_value(displacement, force_smooth)
    # peak_indices = find_insertion_peaks(force_smooth, max_displacement_index, distance=150, height=1.5, prominence=0.5)
    peak_indices = find_insertion_peaks(force_smooth, max_displacement_index, *distance_height_prominence)
    X, y_pred, gradient = linear_regression(displacement, force_smooth, peak_indices, 0.98)

    print(f"Peaks found at indices: {peak_indices}")

    peak_locations = displacement[peak_indices]
    peak_heights = force_smooth[peak_indices]

    peak_1_location, peak_1_index, peak_1_height = (peak_locations[0], peak_indices[0], peak_heights[0]) if len(peak_indices) > 0 else (np.nan, np.nan, np.nan)
    peak_2_location, peak_2_index, peak_2_height = (peak_locations[1], peak_indices[1], peak_heights[1]) if len(peak_indices) > 1 else (np.nan, np.nan, np.nan)

    return X, y_pred, gradient, final_insertion_force, peak_1_index, peak_1_height, peak_2_index, peak_2_height

def calculate_torque_metrics(displacement, force_smooth, distance_height_prominence = ([],[],[])):
    """Calculate metrics from the torque and displacement data."""
    max_displacement_index, final_insertion_torque = get_final_insertion_value(displacement, force_smooth)
    if distance_height_prominence == ([],[],[]):
        peak_indices = find_insertion_peaks(force_smooth, max_displacement_index, distance=150, height=final_insertion_torque / 4, prominence=0.05)
    else:
        peak_indices = find_insertion_peaks(force_smooth, max_displacement_index, *distance_height_prominence)
    retraction_max_index = find_retraction_maxima(force_smooth, max_displacement_index)
    peak_indices = [peak_indices[0] if len(peak_indices) > 0 else np.nan, retraction_max_index]
    
    print(f"Peaks found at indices: {peak_indices}")

    peak_locations = [displacement[int(idx)] if not np.isnan(idx) else np.nan for idx in peak_indices]
    peak_heights = [force_smooth[int(idx)] if not np.isnan(idx) else np.nan for idx in peak_indices]

    peak_1_location, peak_1_index, peak_1_height = (peak_locations[0], peak_indices[0], peak_heights[0]) if len(peak_indices) > 0 else (np.nan, np.nan, np.nan)
    peak_2_location, peak_2_index, peak_2_height = (peak_locations[1], peak_indices[1], peak_heights[1]) if len(peak_indices) > 1 else (np.nan, np.nan, np.nan)

    return final_insertion_torque, peak_1_index, peak_1_height, peak_2_index, peak_2_height

def plot_peaks(peak_1_location, peak_1_height, peak_2_location, peak_2_height, force_name):
    """Highlight and annotate the main peaks in the plot."""
    units = "N" if force_name.startswith('f') else "Nmm"
    peak_name = "Peak 2" if force_name.startswith('f') else "Retraction Max"
    plt.plot(peak_1_location, peak_1_height, "x", color='black')
    plt.annotate(f'Peak 1: {peak_1_height:.1f} {units}', (peak_1_location, peak_1_height), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.plot(peak_2_location, peak_2_height, "x", color='black')
    plt.annotate(f'{peak_name}: {peak_2_height:.1f} {units}', (peak_2_location, peak_2_height), textcoords="offset points", xytext=(0, 10), ha='center')

def plot_regression(X, y_pred, color):
    """Plot the linear regression line."""
    plt.plot(X, y_pred, linestyle='--', color=color)

def plot_config(force_properties, fig_size, x_lim):
    """Configure the plot settings."""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=fig_size)
    plt.title(force_properties['title'])
    plt.ylabel(force_properties['ylabel'])
    plt.ylim(force_properties['ylim'])
    plt.xlabel('Displacement (mm)')
    plt.xlim(x_lim)
    plt.grid(True)

def config_plot_legend(legend_pos, subplot_adjust):
    """Add a legend to the plot and display it."""
    plt.legend(loc='upper right', bbox_to_anchor=legend_pos)
    plt.subplots_adjust(right=subplot_adjust)
    return plt.gcf()

def save_figure(fig, sample_names, force_name, display_metrics, plot_type):
    """Save the figure to a file."""
    save_path = "plot"
    os.makedirs(save_path, exist_ok=True)
    sample_names_str = " ".join(sample_names)
    metrics_text = "with-metrics" if display_metrics else "without-metrics"
    fig_name = os.path.join(save_path, f"{sample_names_str} {force_name} {metrics_text} {plot_type}.png")
    fig.savefig(fig_name, dpi=300)
    print(f"Figure saved as \"{fig_name}\"")

def save_metric_means(sample_names, force_name, mean_sample_data, std_sample_data):
    """Save calculated metrics to a CSV file."""
    save_path = "metrics"
    os.makedirs(save_path, exist_ok=True)
    if force_name.startswith('f'):
        metrics = pd.DataFrame({
            'Sample': [sample.split('_')[0] for sample in sample_names],
            'Velocity (mm/s)': [sample.split('_')[1] for sample in sample_names],
            'Angle (degrees)': [sample.split('_')[2] for sample in sample_names],
            'Needle': [sample.split('_')[3] if len(sample.split('_')) > 3 else '' for sample in sample_names],
            'Mean Gradient (N/mm)': [sample_data[0] for sample_data in mean_sample_data],
            'Std Gradient (N/mm)': [sample_data[0] for sample_data in std_sample_data],
            'Mean Final Insertion Force (N)': [sample_data[1] for sample_data in mean_sample_data],
            'Std Final Insertion Force (N)': [sample_data[1] for sample_data in std_sample_data],
            'Mean Peak 1 Location (mm)': [sample_data[2] for sample_data in mean_sample_data],
            'Std Peak 1 Location (mm)': [sample_data[2] for sample_data in std_sample_data],
            'Mean Peak 1 Height (N)': [sample_data[3] for sample_data in mean_sample_data],
            'Std Peak 1 Height (N)': [sample_data[3] for sample_data in std_sample_data],
            'Mean Peak 2 Location (mm)': [sample_data[4] for sample_data in mean_sample_data],
            'Std Peak 2 Location (mm)': [sample_data[4] for sample_data in std_sample_data],
            'Mean Peak 2 Height (N)': [sample_data[5] for sample_data in mean_sample_data],
            'Std Peak 2 Height (N)': [sample_data[5] for sample_data in std_sample_data]
        })
    else:
        metrics = pd.DataFrame({
            'Sample': [sample.split('_')[0] for sample in sample_names],
            'Velocity (mm/s)': [sample.split('_')[1] for sample in sample_names],
            'Angle (degrees)': [sample.split('_')[2] for sample in sample_names],
            'Needle': [sample.split('_')[3] if len(sample.split('_')) > 3 else '' for sample in sample_names],
            'Mean Final Insertion Torque (Nmm)': [sample_data[1] for sample_data in mean_sample_data],
            'Std Final Insertion Torque (Nmm)': [sample_data[1] for sample_data in std_sample_data],
            'Mean Peak 1 Location (mm)': [sample_data[2] for sample_data in mean_sample_data],
            'Std Peak 1 Location (mm)': [sample_data[2] for sample_data in std_sample_data],
            'Mean Peak 1 Height (Nmm)': [sample_data[3] for sample_data in mean_sample_data],
            'Std Peak 1 Height (Nmm)': [sample_data[3] for sample_data in std_sample_data],
            'Mean Peak 2 Location (mm)': [sample_data[4] for sample_data in mean_sample_data],
            'Std Peak 2 Location (mm)': [sample_data[4] for sample_data in std_sample_data],
            'Mean Peak 2 Height (Nmm)': [sample_data[5] for sample_data in mean_sample_data],
            'Std Peak 2 Height (Nmm)': [sample_data[5] for sample_data in std_sample_data]
        })
    sample_names_str = " ".join(sample_names)
    metrics_name = os.path.join(save_path, f"{sample_names_str} {force_name} mean-metrics.csv")
    metrics.to_csv(metrics_name, index=False)
    print(f"Metrics saved as \"{metrics_name}\"")

def save_metric_single_samples(files, sample_name, force_name, sample_data):
    """Save calculated metrics to a CSV file."""
    sample_numbers = [os.path.basename(file_path).split('_')[4] for file_path in files]
    save_path = "metrics"
    os.makedirs(save_path, exist_ok=True)
    if force_name.startswith('f'):
        metrics = pd.DataFrame({
            'Sample': sample_numbers,
            'Velocity (mm/s)': [sample_name.split('_')[1]] * len(sample_numbers),
            'Angle (degrees)': [sample_name.split('_')[2]] * len(sample_numbers),
            'Needle': [sample_name.split('_')[3]] * len(sample_numbers),
            'Gradient (N/mm)': [sd[0] for sd in sample_data],
            'Final Insertion Force (N)': [sd[1] for sd in sample_data],
            'Peak 1 Location (mm)': [sd[2] for sd in sample_data],
            'Peak 1 Height (N)': [sd[3] for sd in sample_data],
            'Peak 2 Location (mm)': [sd[4] for sd in sample_data],
            'Peak 2 Height (N)': [sd[5] for sd in sample_data]
        })
    else:
        metrics = pd.DataFrame({
            'Sample': sample_numbers,
            'Velocity (mm/s)': [sample_name.split('_')[1]] * len(sample_numbers),
            'Angle (degrees)': [sample_name.split('_')[2]] * len(sample_numbers),
            'Needle': [sample_name.split('_')[3]] * len(sample_numbers),
            'Final Insertion Torque (Nmm)': [sd[1] for sd in sample_data],
            'Peak 1 Location (mm)': [sd[2] for sd in sample_data],
            'Peak 1 Height (Nmm)': [sd[3] for sd in sample_data],
            'Peak 2 Location (mm)': [sd[4] for sd in sample_data],
            'Peak 2 Height (Nmm)': [sd[5] for sd in sample_data]
        })
    metrics_name = os.path.join(save_path, f"{sample_name} {force_name} single-sample-metrics.csv")
    metrics.to_csv(metrics_name, index=False)
    print(f"Metrics saved as \"{metrics_name}\"")

def get_files(data_dir, sample_name):
    """Retrieve the files for a specific sample and force component."""
    files = glob.glob(os.path.join(data_dir, f"{sample_name}*.csv"))
    return files

def get_metadata(sample_name):
    """Retrieve the metadata for a specific sample."""
    parts = sample_name.split('_')
    if len(parts) == 4:
        return tuple(parts)
    elif len(parts) == 3:
        return tuple(parts) + (None,)

def get_metadata_without_needle(sample_name):
    """Retrieve the metadata for a specific sample."""
    sample, velocity, angle = sample_name.split('_')
    return sample, velocity, angle

def trim_and_average_regression_data(X, y_pred):
    """Trim the regression data to the minimum length and average them."""
    X = [x for x in X if len(x) > 0]
    y_pred = [y for y in y_pred if len(y) > 0]
    if len(X) > 0 and len(y_pred) > 0:
        min_length = min(len(x) for x in X)
        X = [x[:min_length] for x in X]
        y_pred = [y[:min_length] for y in y_pred]
        return np.mean(X, axis=0), np.mean(y_pred, axis=0)
    return np.array([]), np.array([])

def filter_and_average_sample_data(sample_data):
    mean_row, std_row = [], []
    for j in range(6):
        valid_values = [sample_data[i][j] for i in range(len(sample_data)) if j < len(sample_data[i]) and not np.isnan(sample_data[i][j])]
        mean_metric = np.mean(valid_values) if len(valid_values) > 0 else np.nan
        std_metric = np.std(valid_values) if len(valid_values) > 0 else np.nan
        mean_row.append(mean_metric)
        std_row.append(std_metric)
    return mean_row, std_row

def filter_and_average_peak_data(sample_data):
    """Filter out NaN values from the peak data."""
    peak_1_locations = [sample_data[i][2] for i in range(len(sample_data)) if not np.isnan(sample_data[i][2])]
    peak_1_location = np.mean(peak_1_locations) if len(peak_1_locations) > 0 else np.nan
    peak_1_heights = [sample_data[i][3] for i in range(len(sample_data)) if not np.isnan(sample_data[i][3])]
    peak_1_height = np.mean(peak_1_heights) if len(peak_1_heights) > 0 else np.nan
    peak_2_locations = [sample_data[i][4] for i in range(len(sample_data)) if not np.isnan(sample_data[i][4])]
    peak_2_location = np.mean(peak_2_locations) if len(peak_2_locations) > 0 else np.nan
    peak_2_heights = [sample_data[i][5] for i in range(len(sample_data)) if not np.isnan(sample_data[i][5])]
    peak_2_height = np.mean(peak_2_heights) if len(peak_2_heights) > 0 else np.nan
    return peak_1_location, peak_1_height, peak_2_location, peak_2_height

def arrange_sample_data(mean_displacement, std_force, gradient, final_insertion_force, peak_1_index, peak_1_height, peak_2_index, peak_2_height):
    """Arrange the sample data into a mean tuple and a std tuple then return."""
    mean_sample_data = (gradient, final_insertion_force, mean_displacement[peak_1_index] if not np.isnan(peak_1_index) else np.nan, peak_1_height, mean_displacement[peak_2_index] if not np.isnan(peak_2_index) else np.nan, peak_2_height)
    std_sample_data = (np.nan, std_force[np.argmax(mean_displacement)], np.nan, std_force[peak_1_index] if not np.isnan(peak_1_index) else np.nan, np.nan, std_force[peak_2_index] if not np.isnan(peak_2_index) else np.nan)
    return mean_sample_data, std_sample_data

def calculate_mean_data_for_plot(displacements, forces):
    """Calculate the mean and standard deviation of the data."""
    mean_displacement = np.mean(displacements, axis=0)
    mean_force = np.mean(forces, axis=0)
    std_force = np.std(forces, axis=0)
    return mean_displacement, mean_force, std_force

def plot_shaded_error(mean_displacement, mean_force, std_force, sample, velocity, angle, color, needle=None):
    """Plot the mean data with shaded error bars."""
    label = f'{sample}\n({velocity} mm/s, {angle}\u00B0, Needle {needle})' if needle else f'{sample}\n({velocity} mm/s, {angle}\u00B0)'
    plt.plot(mean_displacement, mean_force, label=label, color=color)
    plt.fill_between(mean_displacement, mean_force - std_force, mean_force + std_force, alpha=0.8, color=color)

def plot_line(displacement, force, sample_number, color):
    """Plot a line with the specified color."""
    plt.plot(displacement, force, label=sample_number, color=color)

def get_color_from_palette(sample_number):
    """Get a color from the seaborn color palette."""
    color_palette = sns.color_palette()
    return color_palette[sample_number % len(color_palette)]

def print_files(all_files):
    """Print the list of files used for the traces."""
    for i, file in enumerate(all_files, start=1):
        print(f"Files used for trace {i}:\n" + "\n".join([f.split("\\")[-1] for f in file]))
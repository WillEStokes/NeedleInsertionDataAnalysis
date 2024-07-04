import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication
from plotLib import (FORCE_PROPERTY_ARRAY, get_folder_listing, get_sample_names_dialog, get_force_name_dialog, calculate_metrics_dialog, save_figure_dialog, config_plot, config_plot_legend, get_files, load_and_process_data_frames, get_color_from_palette, calculate_force_metrics, calculate_torque_metrics, plot_peaks, plot_regression, plot_line, save_figure, save_metric_single_samples)

DATA_DIRECTORY = "C:\\Users\\menwst\\Documents\\Python\\NeedleInsertionApp\\output\\control needle"
FIGURE_SIZE = (10, 6)
X_LIMIT = (0, 25)
LEGEND_POSITION = (1.6, 1)
SUBPLOT_WIDTH = 0.65
DISTANCE_HEIGHT_PROMINENCE = (200, 1.5, 0.5)

'''Note: Y Limit is set in the FORCE_PROPERTY_ARRAY dictionary for consistency across all plots for each force component'''

def main():
    app = QApplication(sys.argv)

    all_files = get_folder_listing(DATA_DIRECTORY)

    try:
        sample_name = get_sample_names_dialog(all_files, multiple_samples=False)
        force_name = get_force_name_dialog()
    except ValueError as e:
        print(e)
        return

    calculate_metrics_bool = calculate_metrics_dialog()

    print(f"Processing {sample_name}...")

    force_properties = FORCE_PROPERTY_ARRAY[force_name]
    config_plot(force_properties, FIGURE_SIZE, X_LIMIT)

    files = get_files(DATA_DIRECTORY, sample_name)

    pd_data_frames = load_and_process_data_frames(files, force_name, force_properties)

    sample_data = []
    for sample_number, pd_data_frame in pd_data_frames.groupby('Sample Number'):
        sample_number = int(sample_number)
        displacement = pd_data_frame['Displacement']
        force = pd_data_frame['Force']

        displacement = np.array(displacement)
        force = np.array(force)

        color = get_color_from_palette(sample_number)

        if calculate_metrics_bool:
            if force_name.startswith('f'):
                X, y_pred, gradient, final_insertion_force, peak_1_location, peak_1_height, peak_2_location, peak_2_height = calculate_force_metrics(displacement, force, DISTANCE_HEIGHT_PROMINENCE)
            else:
                final_insertion_force, peak_1_location, peak_1_height, peak_2_location, peak_2_height = calculate_torque_metrics(displacement, force)
                gradient = np.nan
            
            plot_peaks(peak_1_location, peak_1_height, peak_2_location, peak_2_height, force_name)
            if force_name.startswith('f'):
                plot_regression(X, y_pred, color)

            sample_data.append((gradient, final_insertion_force, peak_1_location, peak_1_height, peak_2_location, peak_2_height))
        
        plot_line(displacement, force, pd_data_frame['Sample'].iloc[0], color)

    print("Files used:\n" + "\n".join([file.split("\\")[-1] for file in files]))
    fig = config_plot_legend(LEGEND_POSITION, SUBPLOT_WIDTH)
    plt.show()

    if save_figure_dialog():
        save_figure(fig, [sample_name], force_name, calculate_metrics_bool, "single-samples")
        if calculate_metrics_bool:
            save_metric_single_samples(files, sample_name, force_name, sample_data)

    app.quit()

if __name__ == "__main__":
    main()
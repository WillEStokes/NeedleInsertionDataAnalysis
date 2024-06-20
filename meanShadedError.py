import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication
from plotLib import (FORCE_PROPERTY_ARRAY, get_folder_listing, get_folder_listing_without_needle, separate_needles_dialog, get_sample_names_dialog, get_force_name_dialog, calculate_metrics_dialog, plot_config, save_figure_dialog, config_plot_legend, get_files, get_metadata, load_and_process_data, get_color_from_palette, calculate_force_metrics, calculate_torque_metrics, calculate_mean_data_for_plot, trim_and_average_regression_data, plot_peaks, plot_regression, plot_shaded_error, save_figure, save_metric_means, print_files)

DATA_DIRECTORY = "C:\\Users\\menwst\\Documents\\Python\\NeedleInsertionApp\\output"
FIGURE_SIZE = (10, 6)
X_LIMIT= (0, 25)
LEGEND_POSITION = (1.4, 1)
SUBPLOT_ADJUST = 0.7

def main():
    app = QApplication(sys.argv)

    if separate_needles_dialog():
        all_files = get_folder_listing(DATA_DIRECTORY)
    else:
        all_files = get_folder_listing_without_needle(DATA_DIRECTORY)

    try:
        sample_names = get_sample_names_dialog(all_files)
        force_name = get_force_name_dialog()
    except ValueError as e:
        print(e)
        return

    calculate_metrics_bool = calculate_metrics_dialog()

    force_properties = FORCE_PROPERTY_ARRAY[force_name]
    plot_config(force_properties, FIGURE_SIZE, X_LIMIT)

    mean_sample_data, std_sample_data, all_files = [], [], []
    for sample_number, sample_name in enumerate(sample_names):
        print(f"Processing {sample_name}...")

        files = get_files(DATA_DIRECTORY, sample_name)
        sample, velocity, angle, needle = get_metadata(sample_name)

        displacements, forces = load_and_process_data(files, force_name, force_properties)

        color = get_color_from_palette(sample_number)

        if calculate_metrics_bool:
            all_X, all_y_pred, sample_data = [], [], []
            for displacement, force in zip(displacements, forces):
                if force_name.startswith('f'):
                    X, y_pred, gradient, final_insertion_force, peak_1_location, peak_1_height, peak_2_location, peak_2_height = calculate_force_metrics(displacement, force)
                    all_X.append(X)
                    all_y_pred.append(y_pred)
                else:
                    final_insertion_force, peak_1_location, peak_1_height, peak_2_location, peak_2_height = calculate_torque_metrics(displacement, force)
                    gradient = np.nan
                sample_data.append((gradient, final_insertion_force, peak_1_location, peak_1_height, peak_2_location, peak_2_height))

            mean_sample_data.append([np.mean([sample_data[i][j] for i in range(len(sample_data))]) for j in range(6)])
            std_sample_data.append([np.std([sample_data[i][j] for i in range(len(sample_data))]) for j in range(6)])

            plot_peaks(np.mean(sample_data, axis=0)[2], np.mean(sample_data, axis=0)[3], np.mean(sample_data, axis=0)[4], np.mean(sample_data, axis=0)[5])
            if force_name.startswith('f'):
                all_X, all_y_pred = trim_and_average_regression_data(all_X, all_y_pred)
                plot_regression(all_X, all_y_pred, color)

        mean_displacement, mean_force, std_force = calculate_mean_data_for_plot(displacements, forces)

        plot_shaded_error(mean_displacement, mean_force, std_force, sample, velocity, angle, color, needle)
        all_files.append(files)

    print_files(all_files)
    fig = config_plot_legend(LEGEND_POSITION, SUBPLOT_ADJUST)
    plt.show()

    if save_figure_dialog():
        save_figure(fig, sample_names, force_name, calculate_metrics_bool, "shaded-error")
        if calculate_metrics_bool:
            save_metric_means(sample_names, force_name, mean_sample_data, std_sample_data)

    app.quit()

if __name__ == "__main__":
    main()
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication
from plotLib import (FORCE_PROPERTY_ARRAY, get_folder_listing, separate_needles_dialog, get_sample_names_dialog, get_force_name_dialog, calculate_metrics_dialog, plot_config, save_figure_dialog, config_plot_legend, get_files, get_metadata, load_and_process_data, get_color_from_palette, calculate_force_metrics, calculate_torque_metrics, calculate_mean_data_for_plot, trim_and_average_regression_data, filter_sample_data, average_and_arrange_sample_data, plot_peaks, plot_regression, plot_shaded_error, save_figure, save_metric_means, print_files)

DATA_DIRECTORY = "C:\\Users\\menwst\\Documents\\Python\\NeedleInsertionApp\\output"
FIGURE_SIZE = (10, 6)
X_LIMIT= (0, 25)
LEGEND_POSITION = (1.4, 1)
SUBPLOT_WIDTH = 0.7
DISTANCE_HEIGHT_PROMINENCE = (100, 1.5, 0.5)
# DISTANCE_HEIGHT_PROMINENCE = (100, 2, 0.05)

'''Note: Y Limit is set in the FORCE_PROPERTY_ARRAY dictionary for consistency across all plots for each force component'''

def main():
    app = QApplication(sys.argv)

    all_files = get_folder_listing(DATA_DIRECTORY, separate_needles_dialog())

    try:
        sample_names = get_sample_names_dialog(all_files)
        force_name = get_force_name_dialog()
    except ValueError as e:
        print(e)
        return

    calculate_metrics_bool = calculate_metrics_dialog()

    force_properties = FORCE_PROPERTY_ARRAY[force_name]
    plot_config(force_properties, FIGURE_SIZE, X_LIMIT)

    all_mean_sample_data, all_std_sample_data, all_files = [], [], []
    for sample_number, sample_name in enumerate(sample_names):
        print(f"Processing {sample_name}...")

        files = get_files(DATA_DIRECTORY, sample_name)
        sample, velocity, angle, needle = get_metadata(sample_name)

        displacements, forces = load_and_process_data(files, force_name, force_properties)

        color = get_color_from_palette(sample_number)

        mean_displacement, mean_force, std_force = calculate_mean_data_for_plot(displacements, forces)

        if calculate_metrics_bool:
            all_X, all_y_pred, sample_data = [], [], []
            
            '''Calculate metrics on each sample'''
            # for displacement, force in zip(displacements, forces):
            #     if force_name.startswith('f'):
            #         X, y_pred, gradient, final_insertion_force, peak_1_index, peak_1_height, peak_2_index, peak_2_height = calculate_force_metrics(displacement, force, DISTANCE_HEIGHT_PROMINENCE)
            #         all_X.append(X)
            #         all_y_pred.append(y_pred)
            #     else:
            #         final_insertion_force, peak_1_index, peak_1_height, peak_2_index, peak_2_height = calculate_torque_metrics(displacement, force)
            #         gradient = np.nan
            #     sample_data.append((gradient, final_insertion_force, mean_displacement[peak_1_index] if not np.isnan(peak_1_index) else np.nan, peak_1_height, mean_displacement[peak_2_index] if not np.isnan(peak_2_index) else np.nan, peak_2_height))

            # array = filter_sample_data(sample_data)
            # mean_sample_data, std_sample_data = average_and_arrange_sample_data(*array)
            '''End of metrics calculation'''

            '''Calculate metrics on mean data'''
            if force_name.startswith('f'):
                X, y_pred, gradient, final_insertion_force, peak_1_index, peak_1_height, peak_2_index, peak_2_height = calculate_force_metrics(mean_displacement, mean_force, DISTANCE_HEIGHT_PROMINENCE)
                all_X.append(X)
                all_y_pred.append(y_pred)
            else:
                final_insertion_force, peak_1_index, peak_1_height, peak_2_index, peak_2_height = calculate_torque_metrics(mean_displacement, mean_force)
                gradient = np.nan

            mean_sample_data, std_sample_data = average_and_arrange_sample_data([gradient], [final_insertion_force], [mean_displacement[peak_1_index]] if not np.isnan(peak_1_index) else [np.nan], [peak_1_height], [mean_displacement[peak_2_index]] if not np.isnan(peak_2_index) else [np.nan], [peak_2_height], std_force[np.argmax(mean_displacement)], std_force[peak_1_index] if not np.isnan(peak_1_index) else np.nan, std_force[peak_2_index] if not np.isnan(peak_2_index) else np.nan)
            '''End of metrics calculation'''

            all_mean_sample_data.append(mean_sample_data)
            all_std_sample_data.append(std_sample_data)

            plot_peaks(mean_sample_data['Mean Peak 1 Location'], mean_sample_data['Mean Peak 1 Height'], mean_sample_data['Mean Peak 2 Location'], mean_sample_data['Mean Peak 2 Height'], force_name)

            if force_name.startswith('f'):
                all_X, all_y_pred = trim_and_average_regression_data(all_X, all_y_pred)
                plot_regression(all_X, all_y_pred, color)

        plot_shaded_error(mean_displacement, mean_force, std_force, sample, velocity, angle, color, needle)
        all_files.append(files)
    
    print_files(all_files)
    fig = config_plot_legend(LEGEND_POSITION, SUBPLOT_WIDTH)
    plt.show()

    if save_figure_dialog():
        save_figure(fig, sample_names, force_name, calculate_metrics_bool, "shaded-error")
        if calculate_metrics_bool:
            save_metric_means(sample_names, force_name, all_mean_sample_data, all_std_sample_data)

    app.quit()

if __name__ == "__main__":
    main()
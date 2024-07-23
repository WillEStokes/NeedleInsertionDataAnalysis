import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

Y_LIMIT = (0, 20)
SAMPLE_SIZES = [15, 15, 15]
PAIRS = [(0, 1), (1, 2), (0, 2)]

# Set the seaborn theme
sns.set_theme(style="white")

'''Methods'''

def get_paired_t_test_annotations(data, y_name, pairs, sample_sizes):
    annotations = []
    for (i, j) in pairs:
        mean1, std1 = data.iloc[i][f'Mean {y_name}'], data.iloc[i][f'Std {y_name}']
        mean2, std2 = data.iloc[j][f'Mean {y_name}'], data.iloc[j][f'Std {y_name}']
        n1, n2 = sample_sizes[i], sample_sizes[j]
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        se_diff = pooled_std * np.sqrt(1/n1 + 1/n2)
        t_stat = (mean2 - mean1) / se_diff
        df = n1 + n2 - 2
        p_value = stats.t.sf(np.abs(t_stat), df) * 2
        if p_value < 0.001:
            annotations.append('***')
        elif p_value < 0.01:
            annotations.append('**')
        elif p_value < 0.05:
            annotations.append('*')
        else:
            annotations.append('ns')
    return annotations

def add_significance(ax, data, y, pairs, annotations, y_offset=1):
    y_gain = y_offset
    for (group1, group2), annotation in zip(pairs, annotations):
        x1, x2 = group1, group2
        y_max = data[y].max() + y_offset
        ax.plot([x1, x1, x2, x2], [y_max, y_max + y_offset, y_max + y_offset, y_max], lw=1.5, color='k')
        ax.text((x1 + x2) * .5, y_max + y_offset, annotation, ha='center', va='bottom', color='k')
        y_offset+=y_gain

'''Torque Data'''

torque_data = {
    'Sample': ['control-needle (n=15)', 'control-needle-bevel-90 (n=5)', 'novel-needle (n=15)'],
    'Velocity (mm/s)': [2, 2, 2],
    'Angle (degrees)': [0, 0, 0],
    'Peak 1 Load Rate (Nmm/mm)': [None, None, None],
    'Mean Final Insertion Torque (Nmm)': [11.88, 1.41, 10.72],
    'Std Final Insertion Torque (Nmm)': [3.72, 1.12, 2.90],
    'Peak 1 Location (mm)': [15.67, 14.75, 14.68],
    'Mean Peak 1 Height (Nmm)': [7.56, 2.25, 7.39],
    'Std Peak 1 Height (Nmm)': [2.38, 0.67, 2.38],
    'Peak 2 Location (mm)': [18.38, 18.27, 23.29],
    'Mean Peak 2 Height (Nmm)': [18.62, 5.81, 12.33],
    'Std Peak 2 Height (Nmm)': [3.90, 1.18, 2.56],
}

tdf = pd.DataFrame(torque_data)

needle_torque_data = tdf[(tdf['Sample'].isin(['control-needle (n=15)', 'control-needle-bevel-90 (n=5)', 'novel-needle (n=15)'])) & (tdf['Angle (degrees)'] == 0) & (tdf['Velocity (mm/s)'] == 2)]
print(needle_torque_data)

'''Create plot showing the influence of needle tip on Final Insertion Torque'''

fig, ax = plt.subplots(figsize=(6, 4))

# Plot: Influence of needle tip on Final Insertion Torque
y_name = 'Final Insertion Torque (Nmm)'
sns.scatterplot(ax=ax, data=needle_torque_data, x='Sample', y=f'Mean {y_name}', s=100)
ax.errorbar(needle_torque_data['Sample'], needle_torque_data[f'Mean {y_name}'], yerr=needle_torque_data[f'Std {y_name}'], fmt='o')
ax.set_title('Influence of Needle Tip on Final Insertion Torque')
ax.set_xlabel('Sample')
ax.set_ylabel('Final Insertion Torque (Nmm)')
ax.set_ylim(Y_LIMIT)

# Calculate p-values and add significance bars
annotations = get_paired_t_test_annotations(needle_torque_data, y_name, PAIRS, SAMPLE_SIZES)
add_significance(ax, needle_torque_data, f'Mean {y_name}', PAIRS, annotations)

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()

# '''Force Data'''

force_data = {
    'Sample': ['control-needle (n=15)', 'control-needle-bevel-90 (n=5)', 'novel-needle (n=15)'],
    'Velocity (mm/s)': [2, 2, 2],
    'Angle (degrees)': [0, 0, 0],
    'Peak 1 Load Rate (N/mm)': [0.09, 0.30, 0.14],
    'Mean Final Insertion Force (N)': [3.00, 3.60, 0.85],
    'Std Final Insertion Force (N)': [0.64, 0.28, 0.16],
    'Peak 1 Location (mm)': [14.36, 14.33, 10.99],
    'Mean Peak 1 Height (N)': [1.67, 2.84, 0.84],
    'Std Peak 1 Height (N)': [0.52, 0.83, 0.51],
    'Peak 2 Location (mm)': [21.40, 21.86, 22.05],
    'Mean Peak 2 Height (N)': [3.95, 5.09, 5.62],
    'Std Peak 2 Height (N)': [0.87, 0.45, 1.30],
}

fdf = pd.DataFrame(force_data)

needle_force_data = fdf[(fdf['Sample'].isin(['control-needle (n=15)', 'control-needle-bevel-90 (n=5)', 'novel-needle (n=15)'])) & (fdf['Angle (degrees)'] == 0) & (fdf['Velocity (mm/s)'] == 2)]
print(needle_force_data)

'''Create plot showing the influence of needle tip on Peak 2 height'''

fig, ax = plt.subplots(figsize=(6, 4))

# Plot: Influence of needle tip on Peak 2 Height
y_name = 'Peak 2 Height (N)'
sns.scatterplot(ax=ax, data=needle_force_data, x='Sample', y=f'Mean {y_name}', s=100)
ax.errorbar(needle_force_data['Sample'], needle_force_data[f'Mean {y_name}'], yerr=needle_force_data[f'Std {y_name}'], fmt='o')
ax.set_title('Influence of Needle Tip on Peak 2 Height')
ax.set_xlabel('Sample')
ax.set_ylabel('Peak 2 Height (N)')
ax.set_ylim(0, 8)

# Calculate p-values and add significance bars
annotations = get_paired_t_test_annotations(needle_force_data, y_name, PAIRS, SAMPLE_SIZES)
add_significance(ax, needle_force_data, f'Mean {y_name}', PAIRS, annotations, 0.3)

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()
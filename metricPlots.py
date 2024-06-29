import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

Y_LIMIT = (2.5, 6.5)

# Set the seaborn theme
sns.set_theme(style="white")

'''Data'''

data = {
    'Sample': [
        '3mm-artery-filled', '3mm-artery', 'muscle', 'skin-on-3mm-artery', 'skin-on-muscle', 
        'skin-on-muscle', 'skin-on-muscle', 'skin-on-muscle', 'skin-on-muscle', 'skin', 'skin-with-incision'
    ],
    'Velocity (mm/s)': [2, 2, 2, 2, 1, 2, 2, 2, 3, 2, 2],
    'Angle (degrees)': [0, 0, 0, 0, 0, 0, 15, 30, 0, 0, 0],
    'Peak 1 Load Rate (N/mm)': [0.43, 0.33, 0.39, 0.46, 0.46, 0.51, 0.40, 0.22, 0.45, 0.39, 0.31],
    'Mean Final Insertion Force (N)': [1.25, 1.05, 2.20, 2.86, 3.95, 3.49, 4.26, 4.47, 3.20, 1.30, 1.12],
    'Std Final Insertion Force (N)': [0.15, 0.25, 1.54, 0.43, 0.44, 0.62, 0.60, 0.66, 0.64, 0.12, 0.13],
    'Peak 1 Location (mm)': [21.91, 21.81, 23.97, 15.16, 15.60, 15.24, 17.22, 16.68, 14.86, 16.96, 13.93],
    'Mean Peak 1 Height (N)': [2.41, 1.71, 3.32, 4.36, 4.40, 4.93, 4.25, 4.07, 4.06, 4.35, 2.35],
    'Std Peak 1 Height (N)': [0.29, 0.24, 0.73, 0.71, 1.04, 0.33, 0.69, 0.71, 0.78, 0.27, 0.24],
    'Peak 2 Location (mm)': [None, None, None, 21.57, 21.00, 21.16, 24.90, 27.55, 20.40, 19.67, 17.62],
    'Mean Peak 2 Height (N)': [None, None, None, 3.95, 4.76, 4.90, 5.74, 5.40, 4.49, 2.18, 2.38],
    'Std Peak 2 Height (N)': [None, None, None, 0.24, 0.71, 0.60, 0.70, 0.88, 0.84, 0.24, 1.03],
}

'''Data Processing'''

df = pd.DataFrame(data)

# Filter data for angle-related plots for "skin-on-muscle" at velocity 2 mm/s (0, 15, 30 degrees)
angle_data = df[(df['Sample'] == 'skin-on-muscle') & (df['Velocity (mm/s)'] == 2) & (df['Angle (degrees)'].isin([0, 15, 30]))]
print(angle_data)

# Filter data for velocity-related plots for "skin-on-muscle" (1, 2, 3 mm/s)
velocity_data = df[(df['Sample'] == 'skin-on-muscle') & (df['Velocity (mm/s)'].isin([1, 2, 3])) & (df['Angle (degrees)'] == 0)]
print(velocity_data)

# Filter data for artery stiffness-related plots
stiffness_data = df[(df['Sample'].isin(['muscle', '3mm-artery', '3mm-artery-filled'])) & (df['Angle (degrees)'] == 0) & (df['Velocity (mm/s)'] == 2)]
print(stiffness_data)

'''Methods'''

def smooth_line(x, y, degree=2):
    p = np.polyfit(x, y, degree)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_smooth = np.polyval(p, x_smooth)
    return x_smooth, y_smooth

'''Create plot showing the influence of artery stiffness on Peak 1 height'''

fig, ax = plt.subplots(figsize=(6, 4))

# Plot: Influence of artery stiffness on Peak 1 Height
sns.scatterplot(ax=ax, data=stiffness_data, x='Sample', y='Mean Peak 1 Height (N)', s=100)
ax.errorbar(stiffness_data['Sample'], stiffness_data['Mean Peak 1 Height (N)'], yerr=stiffness_data['Std Peak 1 Height (N)'], fmt='o')
ax.set_title('Influence of Artery Stiffness on Peak 1 Height')
ax.set_xlabel('Sample')
ax.set_ylabel('Mean Peak 1 Height (N)')

# Adjust the layout
plt.tight_layout()

# Show the plot
plt.show()

'''Create plots showing the influence of angle on different metrics'''

fig1, axes1 = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Influence of angle on Peak 1 Load Rate
sns.scatterplot(ax=axes1[0, 0], data=angle_data, x='Angle (degrees)', y='Peak 1 Load Rate (N/mm)', s=100)
axes1[0, 0].plot(angle_data['Angle (degrees)'], angle_data['Peak 1 Load Rate (N/mm)'])
# x_smooth, y_smooth = smooth_line(angle_data['Angle (degrees)'], angle_data['Peak 1 Load Rate (N/mm)'])
# axes1[0, 0].plot(x_smooth, y_smooth, color='blue')
axes1[0, 0].set_title('Influence of Angle on Peak 1 Load Rate')
axes1[0, 0].set_xlabel('Angle (degrees)')
axes1[0, 0].set_ylabel('Peak 1 Load Rate (N/mm)')
axes1[0, 0].set_xticks([0, 15, 30])

# Plot 2: Influence of angle on Final Insertion Force
sns.scatterplot(ax=axes1[0, 1], data=angle_data, x='Angle (degrees)', y='Mean Final Insertion Force (N)', s=100)
axes1[0, 1].errorbar(angle_data['Angle (degrees)'], angle_data['Mean Final Insertion Force (N)'], yerr=angle_data['Std Final Insertion Force (N)'], fmt='o')
x_smooth, y_smooth = smooth_line(angle_data['Angle (degrees)'], angle_data['Mean Final Insertion Force (N)'])
axes1[0, 1].plot(x_smooth, y_smooth, color='blue')
axes1[0, 1].set_title('Influence of Angle on Final Insertion Force')
axes1[0, 1].set_xlabel('Angle (degrees)')
axes1[0, 1].set_ylabel('Mean Final Insertion Force (N)')
axes1[0, 1].set_xticks([0, 15, 30])
axes1[0, 1].set_ylim(Y_LIMIT)

# Plot 3: Influence of angle on Peak 1 Height
sns.scatterplot(ax=axes1[1, 0], data=angle_data, x='Angle (degrees)', y='Mean Peak 1 Height (N)', s=100)
axes1[1, 0].errorbar(angle_data['Angle (degrees)'], angle_data['Mean Peak 1 Height (N)'], yerr=angle_data['Std Peak 1 Height (N)'], fmt='o')
x_smooth, y_smooth = smooth_line(angle_data['Angle (degrees)'], angle_data['Mean Peak 1 Height (N)'])
axes1[1, 0].plot(x_smooth, y_smooth, color='blue')
axes1[1, 0].set_title('Influence of Angle on Peak 1 Height')
axes1[1, 0].set_xlabel('Angle (degrees)')
axes1[1, 0].set_ylabel('Mean Peak 1 Height (N)')
axes1[1, 0].set_xticks([0, 15, 30])
axes1[1, 0].set_ylim(Y_LIMIT)

# Plot 4: Influence of angle on Peak 2 Height
sns.scatterplot(ax=axes1[1, 1], data=angle_data, x='Angle (degrees)', y='Mean Peak 2 Height (N)', s=100)
axes1[1, 1].errorbar(angle_data['Angle (degrees)'], angle_data['Mean Peak 2 Height (N)'], yerr=angle_data['Std Peak 2 Height (N)'], fmt='o')
x_smooth, y_smooth = smooth_line(angle_data['Angle (degrees)'], angle_data['Mean Peak 2 Height (N)'])
axes1[1, 1].plot(x_smooth, y_smooth, color='blue')
axes1[1, 1].set_title('Influence of Angle on Peak 2 Height')
axes1[1, 1].set_xlabel('Angle (degrees)')
axes1[1, 1].set_ylabel('Mean Peak 2 Height (N)')
axes1[1, 1].set_xticks([0, 15, 30])
axes1[1, 1].set_ylim(Y_LIMIT)

# Adjust the layout
plt.tight_layout()

# Show the angle-related plots
plt.show()

'''Create plots showing the influence of speed on different metrics'''

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Influence of speed on Peak 1 Load Rate
sns.scatterplot(ax=axes2[0, 0], data=velocity_data, x='Velocity (mm/s)', y='Peak 1 Load Rate (N/mm)', s=100)
axes2[0, 0].plot(velocity_data['Velocity (mm/s)'], velocity_data['Peak 1 Load Rate (N/mm)'])
# x_smooth, y_smooth = smooth_line(velocity_data['Velocity (mm/s)'], velocity_data['Peak 1 Load Rate (N/mm)'])
# axes2[0, 0].plot(x_smooth, y_smooth, color='blue')
axes2[0, 0].set_title('Influence of Speed on Peak 1 Load Rate')
axes2[0, 0].set_xlabel('Velocity (mm/s)')
axes2[0, 0].set_ylabel('Peak 1 Load Rate (N/mm)')
axes2[0, 0].set_xticks([1, 2, 3])

# Plot 2: Influence of speed on Final Insertion Force
sns.scatterplot(ax=axes2[0, 1], data=velocity_data, x='Velocity (mm/s)', y='Mean Final Insertion Force (N)', s=100)
axes2[0, 1].errorbar(velocity_data['Velocity (mm/s)'], velocity_data['Mean Final Insertion Force (N)'], yerr=velocity_data['Std Final Insertion Force (N)'], fmt='o')
x_smooth, y_smooth = smooth_line(velocity_data['Velocity (mm/s)'], velocity_data['Mean Final Insertion Force (N)'])
axes2[0, 1].plot(x_smooth, y_smooth, color='blue')
axes2[0, 1].set_title('Influence of Speed on Final Insertion Force')
axes2[0, 1].set_xlabel('Velocity (mm/s)')
axes2[0, 1].set_ylabel('Mean Final Insertion Force (N)')
axes2[0, 1].set_xticks([1, 2, 3])
axes2[0, 1].set_ylim(Y_LIMIT)

# Plot 3: Influence of speed on Peak 1 Height
sns.scatterplot(ax=axes2[1, 0], data=velocity_data, x='Velocity (mm/s)', y='Mean Peak 1 Height (N)', s=100)
axes2[1, 0].errorbar(velocity_data['Velocity (mm/s)'], velocity_data['Mean Peak 1 Height (N)'], yerr=velocity_data['Std Peak 1 Height (N)'], fmt='o')
x_smooth, y_smooth = smooth_line(velocity_data['Velocity (mm/s)'], velocity_data['Mean Peak 1 Height (N)'])
axes2[1, 0].plot(x_smooth, y_smooth, color='blue')
axes2[1, 0].set_title('Influence of Speed on Peak 1 Height')
axes2[1, 0].set_xlabel('Velocity (mm/s)')
axes2[1, 0].set_ylabel('Mean Peak 1 Height (N)')
axes2[1, 0].set_xticks([1, 2, 3])
axes2[1, 0].set_ylim(Y_LIMIT)

# Plot 4: Influence of speed on Peak 2 Height
sns.scatterplot(ax=axes2[1, 1], data=velocity_data, x='Velocity (mm/s)', y='Mean Peak 2 Height (N)', s=100)
axes2[1, 1].errorbar(velocity_data['Velocity (mm/s)'], velocity_data['Mean Peak 2 Height (N)'], yerr=velocity_data['Std Peak 2 Height (N)'], fmt='o')
x_smooth, y_smooth = smooth_line(velocity_data['Velocity (mm/s)'], velocity_data['Mean Peak 2 Height (N)'])
axes2[1, 1].plot(x_smooth, y_smooth, color='blue')
axes2[1, 1].set_title('Influence of Speed on Peak 2 Height')
axes2[1, 1].set_xlabel('Velocity (mm/s)')
axes2[1, 1].set_ylabel('Mean Peak 2 Height (N)')
axes2[1, 1].set_xticks([1, 2, 3])
axes2[1, 1].set_ylim(Y_LIMIT)

# Adjust the layout
plt.tight_layout()

# Show the speed-related plots
plt.show()
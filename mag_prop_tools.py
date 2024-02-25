##################################################
# BLT_lab_tools.py
# Original author: MSc. Edgar Regulo Vega Carrasco
# Version: 1.0
# Last modification: 2024-11-02
# Description: 
#              1. mag_pro_extractor
#              This script extracts the magnetic properties from a csv file
#              and plots the data and the magnetic properties in a 1x2 grid.
#              The magnetic properties are: Mr, Hc and Ms.
#              The magnetic properties are calculated using linear regression.
#              The area under the curve was not accurate using numerical integration (Simpson's rule).
#              For now, the area under the curve is not calculated.
#
#              2. SAR_extractor
#              This script extracts the SAR from a csv file
##################################################
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import ConvexHull
from scipy.integrate import simps
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import re
import os


def mag_pro_extractor(csv_file_path,csv_image_name,csv_file_name, unit_x,unit_y,min_x, max_x, min_y, max_y):
    """
    
    """
    # Combine the path and filename to create the full file path
    full_csv_file_path = csv_file_path + '/' + csv_file_name
    try:
    # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(full_csv_file_path, names=['x', 'y'])
    except FileNotFoundError:
        print(f"File '{full_csv_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    # Calculate 'Mr'
    data['x_pos'] =data['x'][(data['x'] > 0) & (data['y'] > 0)]
    data['x_neg'] =data['x'][(data['x'] < 0) & (data['y'] > 0)]
    df_case_type = data[(data['y'] > 0)&(data['x'] < 0)]
    case_type = df_case_type['x'].count() 
    Mr_point_1 = data.loc[data['x_neg'] == data['x_neg'].max()]
    Mr_point_2 = data.loc[data['x_pos'] == data['x_pos'].min()]
    Mr_df_points_1_2 = pd.concat([Mr_point_1, Mr_point_2])
    Mr_df_points_1_2 = Mr_df_points_1_2.reset_index(drop=True)
    Mr_df_points_1_2 = Mr_df_points_1_2.drop(columns=['x_neg', 'x_pos'])
    # Create a linear regression model to predict Mr (y when x = 0)
    model_Mr = LinearRegression()
    model_Mr.fit(Mr_df_points_1_2[['x']], Mr_df_points_1_2[['y']])
    slope = model_Mr.coef_[0]
    Mr_predicted = model_Mr.intercept_
    if case_type == 0:
        print(f"Mr: 0 {unit_y}(*)")
    else:
        print(f"Mr: {Mr_predicted[0]:.2f} {unit_y}")
    # Calculate 'Hc'
    data['y_pos'] =data['y'][(data['y'] > 0) & (data['x'] < 0)]
    data['y_neg'] =data['y'][(data['y'] < 0) & (data['x'] < 0)]
    Hc_point_1 = data.loc[data['y_neg'] == data['y_neg'].max()]
    Hc_point_2 = data.loc[data['y_pos'] == data['y_pos'].min()]
    Hc_df_points_1_2 = pd.concat([Hc_point_1, Hc_point_2])
    Hc_df_points_1_2 = Hc_df_points_1_2.reset_index(drop=True)
    Hc_df_points_1_2 = Hc_df_points_1_2.drop(columns=['x_neg', 'x_pos', 'y_neg', 'y_pos'])
    # Create a linear regression model to predict Mr (y when x = 0)
    model_Hc = LinearRegression()
    model_Hc.fit(Hc_df_points_1_2[['x']], Hc_df_points_1_2[['y']])
    slope = model_Hc.coef_[0]
    intercept = model_Hc.intercept_
    # y=slope*x + intercept
    # y=0 when x = -intercept/slope
    # since X values are negative, Hc has to be multiplied by -1 to get a positive value therefore Hc = intercept/slope
    if slope == 0:
        Hc_predicted = 0
        case_type = 0
    else:
        Hc_predicted = intercept/slope
    if case_type == 0:
        print(f"Hc: 0 {unit_x}(*)")
    else:
        print(f"Hc: {Hc_predicted[0]:.2f} {unit_x}")
    # Calculate 'Ms'
    data['y_pos_II'] =data['y'][(data['y'] > 0) & (data['x'] > 0)]
    data['delta_y']=data['y_pos_II'].max()-data['y_pos_II']
    # Calculate the threshold value as 1% of the maximum value in 'abs_y'
    threshold = data['y_pos_II'].max() * 0.01
    row_indices_below_threshold = data[data['delta_y'] < threshold].index.tolist()
    # Extract 'abs_y' values using the row indices
    y_pos_values_below_threshold = data.loc[row_indices_below_threshold, 'y_pos_II'].tolist()
    # Calculate 'Ms' as the average of the 'abs_y' values below the threshold
    Ms_predicted = sum(y_pos_values_below_threshold) / len(y_pos_values_below_threshold)
    # Create a 1x2 grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # Plot the image in the first row
    image_path = csv_file_path + '/' + csv_image_name
    img = plt.imread(image_path)
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title(str(csv_image_name))
    # Create a scatter plot of the data in the second row
    axes[1].scatter(data['x'], data['y'], edgecolor='none', marker='o', s=10, label='Data Points')
    axes[1].set_xlabel(f'Applied Field (H) {unit_x}')
    axes[1].set_ylabel(f'Magnetization (M) {unit_y}')
    axes[1].set_xlim(min_x, max_x)
    axes[1].set_ylim(min_y, max_y)
    axes[1].grid(False)
    # Add a line along the x-axis and y-axis
    axes[1].axhline(0, color='black', linestyle='--', lw=1)
    axes[1].axvline(0, color='black', linestyle='--', lw=1)
    axes[1].set_title(str(csv_file_name))
    Mr_graph = f"{Mr_predicted[0]:.2f} {unit_y}"
    if Hc_predicted == 0:
        Hc_graph = f"0 {unit_x}"
    else:
        Hc_graph = f"{Hc_predicted[0]:.2f} {unit_x}"
    Ms_graph = f"{Ms_predicted:.2f} {unit_y}"
    # Add Hc_predicted, tt
    if case_type == 0:
        axes[1].scatter(0, 0, color='purple', marker='*', s=100, label=f'Hc: 0 {unit_x}(*)')
    else:
        axes[1].scatter((Hc_predicted*-1), 0, color='brown', marker='o', s=50, label=f'Hc: {Hc_graph}')
    # Add Mr_predicted
    if case_type == 0:
        axes[1].scatter(0, 0, color='orange', marker='*', s=100, label=f'Mr: 0 {unit_y} (*)')
    else:
        axes[1].scatter(0, Mr_predicted, color='red', marker='o', s=50, label=f'Mr: {Mr_graph}')
    # Add Ms_predicted
    if len(y_pos_values_below_threshold) > 5:
        axes[1].scatter(0, Ms_predicted, color='green', marker='o', s=50, label=f'Ms: {Ms_graph}')
        axes[1].plot([0, data['x_pos'].max()], [ Ms_predicted, Ms_predicted], color='green', linewidth=2, linestyle='--')
        print(f"Ms: {Ms_predicted:.2f} {unit_y}")
    else:
        axes[1].scatter(0, Ms_predicted, color='red', marker='*', s=100, label=f'Ms: {Ms_graph} (*)')
        axes[1].plot([0, data['x_pos'].max()], [ Ms_predicted, Ms_predicted], color='red', linewidth=2, linestyle='--')
        print(f"Ms: {Ms_predicted:.2f} {unit_y} (*)")
    axes[1].legend()
    if case_type == 0:
        print('(*): Cannot be calculated accurately')
        axes[1].text(0.5, 0.1, '(*): Cannot be calculated accurately', transform=axes[1].transAxes, verticalalignment='top')
    if len(y_pos_values_below_threshold) < 5:
        print('(*): Cannot be calculated accurately')
        axes[1].text(0.5, 0.1, '(*): Cannot be calculated accurately', transform=axes[1].transAxes, verticalalignment='top')
    # Replace '.csv' with '_' in the csv_file_name
    output_image_name = str(csv_file_name).replace('.csv', '_processed.png')
    # Rest of the code remains the same
    output_image_path = csv_file_path 
    output_image_file = output_image_path + '/' + output_image_name 
    fig.savefig(output_image_file, dpi=300, bbox_inches='tight')

def SAR_deltha_T_extractor(csv_file_path,csv_image_name,csv_file_name, unit_x,unit_y,
                  delta_time,heat_capacity_JgC,mass_MNP_g,mass_medium_g,density_medium_gcm3):
    x_label = 'time (' + unit_x + ')'
    y_label = r'$\Delta$ Temperature (' + unit_y + ')'
    # Combine the path and filename to create the full file path
    full_csv_file_path = csv_file_path + '/' + csv_file_name
    def extract_name_column(s):
        # The pattern looks for any sequence of characters between the last "_" and ".csv"
        match = re.search(r'_(?P<content>[^_]+)\.csv$', s)
        if match:
            return match.group('content')
        else:
            return None   
    # Extract the name of the sample from the CSV file name 
    sample_name = extract_name_column(csv_file_name)
    try:
    # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(full_csv_file_path, names=[f'{sample_name}_x', f'{sample_name}_y'])
    except FileNotFoundError:
        print(f"File '{full_csv_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    x_to_SAR = data[f'{sample_name}_x'][data[f'{sample_name}_x']<delta_time]
    x_to_SAR = x_to_SAR.to_numpy()
    y_to_SAR = data[f'{sample_name}_y'][data[f'{sample_name}_x']<delta_time]
    y_to_SAR = y_to_SAR.to_numpy()
    model_SAR = LinearRegression()
    model_SAR.fit(x_to_SAR.reshape(-1, 1), y_to_SAR)
    slope_SAR = model_SAR.coef_[0]
    # Predict y-values using the linear regression model
    y_predicted = model_SAR.predict(x_to_SAR.reshape(-1, 1))
    # Create a 1x2 grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    # Plot the image in the first row
    image_path = csv_file_path + '/' + csv_image_name
    img = plt.imread(image_path)
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title(str(csv_image_name))
    # Create a scatter plot of the data in the second row
    axes[1].plot(data[f'{sample_name}_x'], data[f'{sample_name}_y'], 'o', label=f'{sample_name}_data')
    axes[1].plot(x_to_SAR, y_to_SAR, 'r.', label='data $\Delta$time')
    axes[1].plot(x_to_SAR, y_predicted, 'r-', label='fitted line')
    axes[1].set_xlabel(f'{x_label}')
    axes[1].set_ylabel(f'{y_label}')
    axes[1].grid(False)
    axes[1].set_title(str(csv_file_name))
    axes[1].legend()
    SAR = mass_medium_g*heat_capacity_JgC/mass_MNP_g*slope_SAR
    print(f'SAR = {SAR:.2f} W/g')
    # Add the SAR value to the plot
    sar_text = f"{sample_name} : SAR = {SAR:.2f} W/g"
    axes[1].text(0.7, 0.40, sar_text, transform=axes[1].transAxes, verticalalignment='top') 
    axes[1].text(0.7, 0.35, f'Mass MNP {mass_MNP_g} g', transform=axes[1].transAxes, verticalalignment='top')
    axes[1].text(0.7, 0.30, f'Initial-slope: $\Delta$ time= {delta_time} {unit_x}', transform=axes[1].transAxes, verticalalignment='top')
    axes[1].text(0.7, 0.25, f'Heat Capacity {heat_capacity_JgC} J/g°C', transform=axes[1].transAxes, verticalalignment='top')
    axes[1].text(0.7, 0.20, f'Mass medium {mass_medium_g} g', transform=axes[1].transAxes, verticalalignment='top')
    axes[1].text(0.7, 0.15, f'Density medium {density_medium_gcm3} g/cm$^3$', transform=axes[1].transAxes, verticalalignment='top')
    output_image_name = str(csv_file_name).replace('.csv', '_processed.png')    
    # Rest of the code remains the same
    output_image_path = csv_file_path 
    output_image_file = output_image_path + '/' + output_image_name 
    fig.savefig(output_image_file, dpi=300, bbox_inches='tight')


def SAR_plot_extractor(csv_file_path, csv_image_name, csv_file_name, feature_x, unit_x, feature_y, unit_y,
                      min_x, max_x, min_y, max_y,type_graph):
    """
    
    """
    # Combine the path and filename to create the full file path
    full_csv_file_path = csv_file_path + '/' + csv_file_name
    try:
    # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(full_csv_file_path, names=['x', 'y'])
    except FileNotFoundError:
        print(f"File '{full_csv_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    #round to 2 digits the values of x and y
    data = data.round(2)
    # Create a 1x3 grid of subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    # Plot the image in the first row
    image_path = csv_file_path + '/' + csv_image_name
    img = plt.imread(image_path)
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title(str(csv_image_name))
    # Create a scatter plot of the data in the second row
    if type_graph == 'scatter':
        axes[1].scatter(data['x'], data['y'], color='blue', marker='o', label='Data Points')
    if type_graph == 'line':
        axes[1].plot(data['x'], data['y'], color='blue', linestyle='-', marker='o', markersize=8, label='Data Line')
    if type_graph == 'bar':
        axes[1].bar(data['x'], data['y'], width=max_x/10, color='blue', label='Data Bars')        
    axes[1].set_xlabel(f'{feature_x} {unit_x}')
    axes[1].set_ylabel(f'{feature_y} {unit_y}')
    axes[1].set_title(str(csv_file_name))
    axes[1].set_xlim(min_x, max_x)
    axes[1].set_ylim(min_y, max_y)
    # Extract data for the table
    columns = data.columns
    cell_text = data.values
    # Create the table and add it to the subplot
    labels_table = [f'{feature_x} {unit_x}', f'{feature_y} {unit_y}']
    table = axes[2].table(cellText=cell_text, colLabels=labels_table, loc='center')
    table.auto_set_font_size(False)
    table.auto_set_column_width(col=list(range(len(columns))))  # Set column width based on data
    table.set_fontsize(9)  # Adjust font size as needed
    # Determine the Excel filename
    excel_file_name = csv_file_name.replace('.csv', '.xlsx') 
    # Hide the axes for the table subplot
    axes[2].set_title(excel_file_name)
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()
    # Save the table in our desired path
    excel_file_path = csv_file_path
    # Make sure the directory exists, if not, create it
    if not os.path.exists(excel_file_path):
        os.makedirs(excel_file_path)
    # Create the complete path to save the Excel file
    full_path = os.path.join(excel_file_path, excel_file_name)
    data_excel = data.copy()
    data_excel.columns = [f'{feature_x} {unit_x}', f'{feature_y} {unit_y}']
    # Write DataFrame to Excel at the specified path
    data_excel.to_excel(full_path, engine='openpyxl', index=False)
    # Export the 1x3 grid to an image file
    # Replace '.csv' with '_' in the csv_file_name
    output_image_name = str(csv_file_name).replace('.csv', '_processed.png')
    # Rest of the code remains the same
    output_image_path = csv_file_path 
    output_image_file = output_image_path + '/' + output_image_name 
    fig.savefig(output_image_file, dpi=300, bbox_inches='tight')

def DLS_plot_extractor(csv_file_path, csv_image_name, csv_file_name, feature_x, unit_x, feature_y, unit_y,
                      min_x, max_x, min_y, max_y,type_graph,x_axis_log, peak_detect, peak_start, peak_end):
    """
    
    """
    # Combine the path and filename to create the full file path
    full_csv_file_path = csv_file_path + '/' + csv_file_name
    try:
    # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(full_csv_file_path, names=['x', 'y'])
    except FileNotFoundError:
        print(f"File '{full_csv_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    #Sort data by x values
    data = data.sort_values(by=['x'], ascending=False)
    # Create data with noise for the plot
    data[f'{feature_x}_{unit_x}'] = data['x']
    data[f'{feature_y}_{unit_y}'] = data['y'].apply(lambda x: max(0,x))
    # Ensure that the intensities sum to 1 (or very close to 1)
    data[f'{feature_y}_{unit_y}_normalized'] = data[f'{feature_y}_{unit_y}']/ data[f'{feature_y}_{unit_y}'].sum()
    # Create a 1x2 grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # Plot the image in the first row
    image_path = csv_file_path + '/' + csv_image_name
    img = plt.imread(image_path)
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title(str(csv_image_name))
    # Create a scatter plot of the data in the second row
    if type_graph == 'scatter':
        axes[1].scatter(data['x'], data['y'], color='blue', marker='o', label='Data Points')
    if type_graph == 'line':
        axes[1].plot(data['x'], data['y'], color='blue', linestyle='-', marker='o', markersize=8, label='Data Line')
    if x_axis_log == True:
        axes[1].set_xscale('log')
    # setting the limits for x-axis and y-axis
    axes[1].set_xlabel(f'{feature_x} {unit_x}')
    axes[1].set_ylabel(f'{feature_y} {unit_y}')
    axes[1].set_title(str(csv_file_name))
    axes[1].set_xlim(min_x, max_x)
    axes[1].set_ylim(min_y, max_y)    
    y_noise_threshold = data['y'].max() * 0.001
    if peak_detect == 'auto':
        data_wo_noise = data[data['y'] > y_noise_threshold]
    if peak_detect == 'manual':
        data_wo_noise = data[(data['x'] > peak_start) & (data['x'] < peak_end)]
    #Sort data_wo_noise by x values
    data_wo_noise = data_wo_noise.sort_values(by=['x'], ascending=False)
    # Ensure that the intensities sum to 1 (or very close to 1)
    average_size = (data_wo_noise[f'{feature_x}_{unit_x}'] * data_wo_noise[f'{feature_y}_{unit_y}_normalized']).sum()
    # Calculate the weighted variance, weighted standard deviation, max size, max y feature and PDI
    #weighted_variance = ((data_wo_noise[f'{feature_x}_{unit_x}'] - average_size)**2 * data_wo_noise[f'{feature_y}_{unit_y}_normalized']).sum()
    n_weights = data_wo_noise[f'{feature_y}_{unit_y}_normalized'].count()
    weighted_variance = (((data_wo_noise[f'{feature_x}_{unit_x}'] - average_size)**2 * data_wo_noise[f'{feature_y}_{unit_y}_normalized']).sum())/(((data_wo_noise[f'{feature_y}_{unit_y}_normalized']).sum())*((n_weights-1)/n_weights))
    
    weighted_std_dev= np.sqrt(weighted_variance)
    max_size = data.loc[data_wo_noise[f'{feature_y}_{unit_y}'].idxmax(), f'{feature_x}_{unit_x}']
    max_y_feature = data_wo_noise[f'{feature_y}_{unit_y}'].max()
    PDI = (weighted_std_dev/average_size)**2
    axes[1].text(0.05, 0.95, f'Average diameter ($D_{{avg}}$): {average_size:.2f} {unit_x}', transform=axes[1].transAxes, verticalalignment='top') 
    axes[1].text(0.05, 0.90, f'Weighted standard deviation ($std_{{w}}$): {weighted_std_dev:.2f} {unit_x}', transform=axes[1].transAxes, verticalalignment='top') 
    axes[1].text(0.05, 0.85, f'Diameter at max {feature_y} ({max_y_feature:.2f}{unit_y}): {max_size:.2f} {unit_x}' , transform=axes[1].transAxes, verticalalignment='top')
    axes[1].text(0.05, 0.80, f'Polydispersity Index PDI=$(std_{{w}}/D_{{avg}})^2$: {PDI:.4f}' , transform=axes[1].transAxes, verticalalignment='top') 
    #plot the data without noise
    axes[1].plot(data_wo_noise[f'{feature_x}_{unit_x}'], data_wo_noise[f'{feature_y}_{unit_y}'],
                 color='red', linestyle='-', marker='o', label='Data for calculations')
    axes[1].legend()
    print(f'Average diameter: {average_size:.2f} {unit_x}')
    print(f'Weighted standard deviation: {weighted_std_dev:.2f} {unit_x}')
    print(f'Diameter at max {feature_y} ({max_y_feature:.2f}{unit_y}): {max_size:.2f} {unit_x}')
    print(f'Polydispersity Index (PDI): {PDI:.4f}')
    plt.tight_layout()
    plt.show()
    # Export the 1x2 grid to an image file
    # Replace '.csv' with '_' in the csv_file_name
    output_image_name = str(csv_file_name).replace('.csv', '_processed.png')

    # Rest of the code remains the same
    output_image_path = csv_file_path 
    output_image_file = output_image_path + '/' + output_image_name 
    fig.savefig(output_image_file, dpi=300, bbox_inches='tight')


def SAR_Bekovic_Hamler_extractor(csv_file_path,csv_image_name,csv_file_name,feature_x, unit_x,feature_y,unit_y,min_x, max_x, min_y, max_y,
                                 heat_capacity_JgC,mass_MNP_g,mass_medium_g):
    # Combine the path and filename to create the full file path
    full_csv_file_path = csv_file_path + '/' + csv_file_name
    try:
    # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(full_csv_file_path, names=['x', 'y'])
    except FileNotFoundError:
        print(f"File '{full_csv_file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    #Get labels
    if feature_y == 'delta_temperature':
        y_label = r'$\Delta$ Temperature (' + unit_y + ')'
    if feature_y == 'temperature':
        y_label = 'Temperature (' + unit_y + ')'
    # create x_label
    x_label = 'time (' + unit_x + ')'
    # filter negative values of x and y due to exctraction errors
    data = data[data['x'] >= 0]
    data = data[data['y'] >= 0]
    # sort values by x
    data = data.sort_values(by=['x'])
    # reset index
    data = data.reset_index(drop=True)
    data[f'{feature_x}_{unit_x}'] = data["x"]
    data[f'{feature_y}_{unit_y}'] = data["y"]
    if feature_y == 'delta_temperature':
        data['delta_temperature'] = data["y"]
    if feature_y == 'temperature':
        data['delta_temperature'] = data["y"] - data["y"].iloc[0]
    # get initial temperature and delta temperature
    temperature_initial = data[f'{feature_y}_{unit_y}'].iloc[0]
    delta_temperature_max = data["delta_temperature"].max()
    temperature_max = data[f'{feature_y}_{unit_y}'].max()
    # initial guess for tau
    tau_guess = (data[f'{feature_x}_{unit_x}'].max())/5
    # initial slope function
    def bekovic_hamler_initial_slope(time, tau, temperature_initial, delta_temperature_max):
        return temperature_initial + delta_temperature_max * (1 - np.exp(-time / tau))
    # curve fit
    popt, pcov = curve_fit(bekovic_hamler_initial_slope, data[f'{feature_x}_{unit_x}'], data['delta_temperature'], p0=[tau_guess, temperature_initial, delta_temperature_max])
    # Fitted parameters
    tau_fitted = popt[0]
    time_min = data[f'{feature_x}_{unit_x}'].iloc[0]
    # Using the fitted tau to calculate the temperature at any given time
    time_values = np.linspace(data[f'{feature_x}_{unit_x}'].min(), data[f'{feature_x}_{unit_x}'].max(), 100)
    fitted_temperatures = bekovic_hamler_initial_slope(time_values, tau_fitted, temperature_initial, delta_temperature_max)
    # Calculate the initial slope
    initial_slope_unit_yx = delta_temperature_max/tau_fitted
    # Create a 1x2 grid of subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    # Plot the image in the first row
    image_path = csv_file_path + '/' + csv_image_name
    img = plt.imread(image_path)
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title(str(csv_image_name))
    # Create a scatter plot of the data in the second row
    # Plot the fitted curve:

    if unit_x == 'min': 
        initial_slope_unit_yseg = initial_slope_unit_yx / 60
    if unit_x == 'seg': 
        initial_slope_unit_yseg = initial_slope_unit_yx
    
    slope_SAR = initial_slope_unit_yseg
    SAR = mass_medium_g*heat_capacity_JgC/mass_MNP_g*slope_SAR

    if feature_y == 'delta_temperature':
        axes[1].scatter(data[f'{feature_x}_{unit_x}'], data['delta_temperature'], label='Data')
        axes[1].plot(time_values, fitted_temperatures, label='Bekovic-Hamler fitted curve', color='red')
        axes[1].plot([0, tau_fitted], [temperature_initial, delta_temperature_max], label=f'Slope ($\Delta T_{{max}}/\\tau$): {initial_slope_unit_yx:.2f} {unit_y}/{unit_x}', color='green', linewidth=2, linestyle='--')
    if feature_y == 'temperature':
        axes[1].scatter(data[f'{feature_x}_{unit_x}'], data[f'{feature_y}_{unit_y}'], label='Data')
        axes[1].plot(time_values, fitted_temperatures, label='Bekovic-Hamler fitted curve', color='red')
        axes[1].plot([time_min, tau_fitted], [temperature_initial, temperature_max], label=f'Slope ($\Delta T_{{max}}/\\tau$): {initial_slope_unit_yx:.2f} {unit_y}/{unit_x}', color='green', linewidth=2, linestyle='--')
    axes[1].set_title(str(csv_file_name))
    axes[1].set_xlim(min_x, max_x)
    axes[1].set_ylim(min_y, max_y)  
    axes[1].text(0.05, 0.95, f'SAR = {SAR:.2f} W/g', transform=axes[1].transAxes, verticalalignment='top') 
    axes[1].text(0.05, 0.90, f'Fitted tau($\\tau$)= {tau_fitted:.2f} {unit_x}', transform=axes[1].transAxes, verticalalignment='top') 
    axes[1].text(0.05, 0.85, f'Heat Capacity = {heat_capacity_JgC} J/gC', transform=axes[1].transAxes, verticalalignment='top') 
    axes[1].text(0.05, 0.80, f'Mass MNP = {heat_capacity_JgC} g', transform=axes[1].transAxes, verticalalignment='top') 
    axes[1].text(0.05, 0.75, f'Mass medium = {mass_medium_g} g', transform=axes[1].transAxes, verticalalignment='top') 
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel(y_label)
    axes[1].legend(loc = 'lower right')
    plt.tight_layout()
    plt.show()
    print('Fitted tau =', f'{tau_fitted:.2f} {unit_x}')
    print('Initial Slope:', f'{initial_slope_unit_yx:.2f} {unit_y}/{unit_x}')
    print(f'SAR = {SAR:.2f} W/g')
    # Export the 1x2 grid to an image file
    # Replace '.csv' with '_' in the csv_file_name
    output_image_name = str(csv_file_name).replace('.csv', '_processed.png')
    output_image_path = csv_file_path 
    output_image_file = output_image_path + '/' + output_image_name 
    fig.savefig(output_image_file, dpi=300, bbox_inches='tight')

import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

#Fz,Tx,Ty,or Tz
test_type = 'Tz'

if(test_type == 'Fz'):
    voltage_array = []
    mean_Fz_force = []
    with open('fz_data.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) 
        for row in reader:
            voltage_array.append(float(row[0]))
            mean_Fz_force.append(float(row[1]))

    slope, intercept, r_value, p_value, std_err = stats.linregress(voltage_array, mean_Fz_force)
    regression_line = slope * np.array(voltage_array) + intercept

    plt.scatter(voltage_array, mean_Fz_force, marker='o', s=100,label='$F_z$')
    plt.plot(voltage_array, regression_line, color='red', label='Linear Regression')
    plt.xticks(np.arange(5, 15.5, 1))
    plt.text(5, 0.05, r'$F_G=-0.1095N$', fontsize=12)

    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=-intercept/slope, color='gray', linestyle='--')
    plt.text(-intercept/slope+0.35, -0.07, '{:.1f}'.format(-intercept/slope), fontsize=10, ha='right')
    plt.xlabel('Voltage(V)')
    plt.ylabel('$F_z$ (N)')
    plt.title('Fz')
    plt.legend()
    plt.show()

if(test_type == 'Tx'):
    differential_voltage_array = []
    mean_Tx = []
    with open('Tx_data.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) 
        for row in reader:
            differential_voltage_array.append(float(row[0]))
            mean_Tx.append(float(row[1]))

    slope, intercept, r_value, p_value, std_err = stats.linregress(differential_voltage_array, mean_Tx)
    regression_line = slope * np.array(differential_voltage_array) + intercept

    plt.scatter(differential_voltage_array[::2], mean_Tx[::2], marker='o', s=100,label='roll torque')
    plt.plot(differential_voltage_array, regression_line, color='red', label='Linear Regression')
    plt.xticks(np.arange(-2, 2.2, 0.2))

    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=-intercept/slope, color='gray', linestyle='--')
    plt.text(-intercept/slope+0.35, -0.07, '{:.1f}'.format(-intercept/slope), fontsize=10, ha='right')
    plt.xlabel('differential voltage(V)')
    plt.ylabel('roll torque (N $\\cdot$ m)')
    plt.title('Tx')
    plt.legend()
    plt.show()

if(test_type == 'Ty'):
    mean_voltage_array = []
    mean_Ty = []
    with open('Ty_data.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) 
        for row in reader:
            mean_voltage_array.append(float(row[0]))
            mean_Ty.append(float(row[1]))

    slope, intercept, r_value, p_value, std_err = stats.linregress(mean_voltage_array, mean_Ty)
    regression_line = slope * np.array(mean_voltage_array) + intercept

    plt.scatter(mean_voltage_array[::2], mean_Ty[::2], marker='o', s=100,label='pitch torque')
    plt.plot(mean_voltage_array, regression_line, color='red', label='Linear Regression')
    plt.xticks(np.arange(-3, 3.2, 0.2))

    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=-intercept/slope, color='gray', linestyle='--')

    plt.xlabel('mean voltage(V)')
    plt.ylabel('pitch torque (N $\\cdot$ m)')
    plt.title('Ty')
    plt.legend()
    plt.show()

if(test_type == 'Tz'):
    split_cycle_array = []
    mean_Tz = []
    with open('Tz_data.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader) 
        for row in reader:
            split_cycle_array.append(float(row[0]))
            mean_Tz.append(float(row[1]))

    slope, intercept, r_value, p_value, std_err = stats.linregress(split_cycle_array, mean_Tz)
    regression_line = slope * np.array(split_cycle_array) + intercept

    plt.scatter(split_cycle_array[::1], mean_Tz[::1], marker='o', s=100,label='$pitch torque$')
    plt.plot(split_cycle_array, regression_line, color='red', label='Linear Regression')
    plt.xticks(np.arange(-0.15, 0.16, 0.01))
    
    plt.axhline(y=0, color='gray', linestyle='--')
    plt.axvline(x=-intercept/slope, color='gray', linestyle='--')
    plt.xlabel('delta split cycle')
    plt.ylabel('pitch torque (N $\\cdot$ m)')
    plt.title('Tz')
    plt.legend()
    plt.show()

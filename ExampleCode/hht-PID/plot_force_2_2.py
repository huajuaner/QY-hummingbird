'''This file will plot mean Fz,Tx,Ty,Tz varying with Ua,Ur,Up,deita sigma.
The data must be generated in test_force.py first.'''
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 创建一个2x2的子图

# Fz
voltage_array = []
mean_Fz_force = []
with open('fz_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        voltage_array.append(float(row[0]))
        mean_Fz_force.append(float(row[1]))

slope, intercept, _, _, _ = stats.linregress(voltage_array, mean_Fz_force)
regression_line = slope * np.array(voltage_array) + intercept

axs[0, 0].scatter(voltage_array, mean_Fz_force, marker='o', s=100, label='$F_z$',color='skyblue')
axs[0, 0].plot(voltage_array, regression_line, color='red')
axs[0, 0].set_xticks(np.arange(5, 15.5, 1))
axs[0, 0].tick_params(axis='x', labelsize=15)
axs[0, 0].set_yticks(np.arange(-0.05, 0.125, 0.025))
axs[0, 0].tick_params(axis='y', labelsize=15)
axs[0, 0].text(5, 0.06, r'$F_G=-0.1095N$', fontsize=20)
axs[0, 0].axhline(y=0, color='gray', linestyle='--')
axs[0, 0].axvline(x=-intercept/slope, color='gray', linestyle='--')
axs[0, 0].text(-intercept/slope + 0.35, -0.06, '{:.1f}'.format(-intercept/slope), fontsize=15, ha='right')
axs[0, 0].set_xlabel('$U_a$(V)',fontsize=20)
axs[0, 0].set_ylabel('$F_z$ (N)',fontsize=20)
# axs[0, 0].set_title('Fz')
axs[0, 0].legend(fontsize=20)

# Tx
differential_voltage_array = []
mean_Tx = []
with open('Tx_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        differential_voltage_array.append(float(row[0]))
        mean_Tx.append(float(row[1]))

slope, intercept, _, _, _ = stats.linregress(differential_voltage_array, [x * 1000 for x in mean_Tx])
regression_line = slope * np.array(differential_voltage_array) + intercept

axs[0, 1].scatter(differential_voltage_array,[x * 1000 for x in mean_Tx], marker='o', s=100, label='roll torque',color='thistle')
axs[0, 1].plot(differential_voltage_array, regression_line, color='red')
axs[0, 1].set_xticks(np.arange(-2, 2.2, 0.4))
axs[0, 1].tick_params(axis='x', labelsize=15)
axs[0, 1].set_yticks(np.arange(-2.5, 3, 0.5))
axs[0, 1].tick_params(axis='y', labelsize=15)
axs[0, 1].axhline(y=0, color='gray', linestyle='--')
axs[0, 1].axvline(x=-intercept/slope, color='gray', linestyle='--')
axs[0, 1].set_xlabel('$U_d$(V)',fontsize=20)
axs[0, 1].set_ylabel('roll torque (N $\\cdot$ mm)',fontsize=20)
# axs[0, 1].set_title('Tx')
axs[0, 1].legend(fontsize=20)

# Ty
mean_voltage_array = []
mean_Ty = []
with open('Ty_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        mean_voltage_array.append(float(row[0]))
        mean_Ty.append(float(row[1]))

slope, intercept, _, _, _ = stats.linregress(mean_voltage_array, [x * 1000 for x in mean_Ty])
regression_line = slope * np.array(mean_voltage_array) + intercept

axs[1, 0].scatter(mean_voltage_array, [x * 1000 for x in mean_Ty], marker='o', s=100, label='pitch torque',color='sandybrown')
axs[1, 0].plot(mean_voltage_array, regression_line, color='red')
axs[1, 0].set_xticks(np.arange(-3, 3.2, 0.6))
axs[1, 0].tick_params(axis='x', labelsize=15)
axs[1, 0].tick_params(axis='y', labelsize=15)
axs[1, 0].axhline(y=0, color='gray', linestyle='--')
axs[1, 0].axvline(x=-intercept/slope, color='gray', linestyle='--')
axs[1, 0].set_xlabel('$U_p$(V)',fontsize=20)
axs[1, 0].set_ylabel('pitch torque (N $\\cdot$ mm)',fontsize=20)
# axs[1, 0].set_title('Ty')
axs[1, 0].legend(fontsize=20)

# Tz
split_cycle_array = []
mean_Tz = []
with open('Tz_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        split_cycle_array.append(float(row[0]))
        mean_Tz.append(float(row[1]))

slope, intercept, _, _, _ = stats.linregress(split_cycle_array, [x * 1000 for x in mean_Tz])
regression_line = slope * np.array(split_cycle_array) + intercept

axs[1, 1].scatter(split_cycle_array, [x * 1000 for x in mean_Tz], marker='o', s=100, label='$yaw\ torque$',color='lightgreen')
axs[1, 1].plot(split_cycle_array, regression_line, color='red')
axs[1, 1].set_xticks(np.arange(-0.2, 0.21, 0.04))
axs[1, 1].tick_params(axis='x', labelsize=12)
axs[1, 1].tick_params(axis='y', labelsize=15)
axs[1, 1].axhline(y=0, color='gray', linestyle='--')
axs[1, 1].axvline(x=-intercept/slope, color='gray', linestyle='--')
axs[1, 1].set_xlabel('$\Delta \sigma$',fontsize=20)
axs[1, 1].set_ylabel('yaw torque (N $\\cdot$ mm)',fontsize=20)
# axs[1, 1].set_title('Tz')
axs[1, 1].legend(fontsize=20)

plt.tight_layout()  # 调整子图参数以给定指定的填充
plt.show()

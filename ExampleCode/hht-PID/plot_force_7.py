'''Compare to plot_force_2*2.py, this file will plot mean Tx,Ty,Tz varying with Ua '''
import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

fig, axs = plt.subplots(2, 4, figsize=(12, 6))  # 创建一个2x2的子图
legendsize = 16
cornersize = 18
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
axs[0, 0].axhline(y=0, color='gray', linestyle='--')
axs[0, 0].axvline(x=-intercept/slope, color='gray', linestyle='--')
axs[0, 0].text(-intercept/slope + 0.35, -0.065, '{:.1f}'.format(-intercept/slope), fontsize=15, ha='right')
axs[0, 0].set_xlabel('$U_a$(V)',fontsize=20)
axs[0, 0].set_ylabel('$F_z$ (N)',fontsize=20)
# axs[0, 0].set_title('Fz')
# axs[0, 0].text(5, 0.065, r'$F_G=-0.1095N$', fontsize=18)
xlim = axs[0, 0].get_xlim()
ylim = axs[0, 0].get_ylim()
x_pos = xlim[1]
y_pos = 0.95* ylim[0]
axs[0, 0].text(x_pos, y_pos, r'$U_r=0$'+'\n'+r'$U_p=0$'+'\n'+r'$\Delta \sigma=0$', fontsize=cornersize,ha='right')
axs[0, 0].legend(fontsize=legendsize, loc='upper left')

# #Amp
# voltage_array = []
# amp_max = []
# with open('amp_data.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile)
#     next(reader)
#     for row in reader:
#         voltage_array.append(float(row[0]))
#         amp_max.append(float(row[1]))

# slope, intercept, _, _, _ = stats.linregress(voltage_array, amp_max)
# regression_line = slope * np.array(voltage_array) + intercept

# axs[1, 0].scatter(voltage_array, amp_max, marker='o', s=100, label='$F_z$',color='skyblue')
# axs[1, 0].plot(voltage_array, regression_line, color='red')
# axs[1, 0].set_xticks(np.arange(5, 15.5, 1))
# axs[1, 0].tick_params(axis='x', labelsize=15)
# # axs[0, 0].set_yticks(np.arange(-0.05, 0.125, 0.025))
# axs[1, 0].tick_params(axis='y', labelsize=15)
# # axs[1, 0].axhline(y=0, color='gray', linestyle='--')
# # axs[1, 0].axvline(x=-intercept/slope, color='gray', linestyle='--')
# # axs[1, 0].text(-intercept/slope + 0.35, -0.06, '{:.1f}'.format(-intercept/slope), fontsize=15, ha='right')
# axs[1, 0].set_xlabel('$U_a$(V)',fontsize=20)
# axs[1, 0].set_ylabel('$Amp$ (N)',fontsize=20)
# # axs[1, 0].set_title('Fz')
# axs[1, 0].legend(fontsize=20)

axs[1, 0].axis('off')

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
axs[0, 1].set_xticks(np.arange(-2, 2.1, 1))
axs[0, 1].tick_params(axis='x', labelsize=15)
axs[0, 1].set_yticks(np.arange(-2.5, 3, 0.5))
axs[0, 1].tick_params(axis='y', labelsize=15)
axs[0, 1].axhline(y=0, color='gray', linestyle='--')
axs[0, 1].axvline(x=-intercept/slope, color='gray', linestyle='--')
axs[0, 1].set_xlabel('$U_r$(V)',fontsize=20)
axs[0, 1].set_ylabel('roll torque (N $\\cdot$ mm)',fontsize=20)
# axs[0, 1].set_title('Tx')
xlim = axs[0, 1].get_xlim()
ylim = axs[0, 1].get_ylim()
x_pos = xlim[1]
y_pos = 0.95* ylim[0]
axs[0, 1].text(x_pos, y_pos, r'$U_a=10V$'+'\n'+r'$U_p=0$'+'\n'+r'$\Delta \sigma=0$', fontsize=cornersize,ha='right')
axs[0, 1].legend(fontsize=legendsize)

# TxUa
voltage_array = []
mean_TxUa = []
with open('TxUa_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        voltage_array.append(float(row[0]))
        mean_TxUa.append(float(row[1]))

axs[1, 1].scatter(voltage_array, [x * 1000 for x in mean_TxUa], marker='o', s=100, label='roll torque', color='thistle')
axs[1, 1].set_xticks(np.arange(5, 15.5, 1))
axs[1, 1].tick_params(axis='x', labelsize=15)
axs[1, 1].set_yticks(np.arange(-2.5, 3, 0.5))
axs[1, 1].tick_params(axis='y', labelsize=15)
axs[1, 1].set_xlabel('$U_a$(V)', fontsize=20)
axs[1, 1].set_ylabel('roll torque (N $\\cdot$ mm)', fontsize=20)
xlim = axs[1, 1].get_xlim()
ylim = axs[1, 1].get_ylim()
x_pos = xlim[1]
y_pos = 0.95* ylim[0]
axs[1, 1].text(x_pos, y_pos, r'$U_r=1V$'+'\n'+r'$U_p=0$'+'\n'+r'$\Delta \sigma=0$', fontsize=cornersize,ha='right')
axs[1, 1].legend(fontsize=legendsize, loc='upper left')

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

axs[0, 2].scatter(mean_voltage_array, [x * 1000 for x in mean_Ty], marker='o', s=100, label='pitch torque', color='sandybrown')
axs[0, 2].plot(mean_voltage_array, regression_line, color='red')
axs[0, 2].set_xticks(np.arange(-3, 3.2, 1))
axs[0, 2].tick_params(axis='x', labelsize=15)
axs[0, 2].set_yticks(np.arange(-0.75, 0.76, 0.25))
axs[0, 2].tick_params(axis='y', labelsize=15)
axs[0, 2].axhline(y=0, color='gray', linestyle='--')
axs[0, 2].axvline(x=-intercept/slope, color='gray', linestyle='--')
axs[0, 2].set_xlabel('$U_p$(V)', fontsize=20)
axs[0, 2].set_ylabel('pitch torque (N $\\cdot$ mm)', fontsize=20)
# axs[0, 2].set_title('Ty')
xlim = axs[0, 2].get_xlim()
ylim = axs[0, 2].get_ylim()
x_pos = xlim[1]
y_pos = 0.95* ylim[0]
axs[0, 2].text(x_pos, y_pos, r'$U_a=10V$'+'\n'+r'$U_r=0$'+'\n'+r'$\Delta \sigma=0$', fontsize=cornersize,ha='right')
axs[0, 2].legend(fontsize=legendsize, loc='upper left')

# TyUa
voltage_array = []
mean_TyUa = []
with open('TyUa_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        voltage_array.append(float(row[0]))
        mean_TyUa.append(float(row[1]))

slope, intercept, _, _, _ = stats.linregress(voltage_array, [x * 1000 for x in mean_TyUa])
regression_line = slope * np.array(voltage_array) + intercept

axs[1, 2].scatter(voltage_array, [x * 1000 for x in mean_TyUa], marker='o', s=100, label='pitch torque', color='sandybrown')
axs[1, 2].plot(voltage_array, regression_line, color='red')
axs[1, 2].set_xticks(np.arange(5, 15.5, 1))
axs[1, 2].tick_params(axis='x', labelsize=15)
axs[1, 2].set_yticks(np.arange(-0.75, 0.76, 0.25))
axs[1, 2].tick_params(axis='y', labelsize=15)
axs[1, 2].set_xlabel('$U_a$(V)', fontsize=20)
axs[1, 2].set_ylabel('pitch torque (N $\\cdot$ mm)', fontsize=20)
xlim = axs[1, 2].get_xlim()
ylim = axs[1, 2].get_ylim()
x_pos = xlim[1]
y_pos = 0.95* ylim[0]
axs[1, 2].text(x_pos, y_pos, r'$U_r=0$'+'\n'+r'$U_p=1.5V$'+'\n'+r'$\Delta \sigma=0$', fontsize=cornersize,ha='right')
axs[1, 2].legend(fontsize=legendsize, loc='upper left')

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

axs[0, 3].scatter(split_cycle_array, [x * 1000 for x in mean_Tz], marker='o', s=100, label='$yaw\ torque$', color='lightgreen')
axs[0, 3].plot(split_cycle_array, regression_line, color='red')
axs[0, 3].set_xticks(np.arange(-0.2, 0.21, 0.1))
axs[0, 3].tick_params(axis='x', labelsize=12)
axs[0, 3].set_yticks(np.arange(-0.5, 0.51, 0.1))
axs[0, 3].tick_params(axis='y', labelsize=15)
axs[0, 3].axhline(y=0, color='gray', linestyle='--')
axs[0, 3].axvline(x=-intercept/slope, color='gray', linestyle='--')
axs[0, 3].set_xlabel('$\Delta \sigma$', fontsize=20)
axs[0, 3].set_ylabel('yaw torque (N $\\cdot$ mm)', fontsize=20)
# axs[0, 3].set_title('Tz')
xlim = axs[0, 3].get_xlim()
ylim = axs[0, 3].get_ylim()
x_pos = xlim[1]
y_pos = 0.95* ylim[0]
axs[0, 3].text(x_pos, y_pos, r'$U_a=10V$'+'\n'+'$U_r=0$'+'\n'+r'$U_p=0$', fontsize=cornersize,ha='right')
axs[0, 3].legend(fontsize=legendsize)

# TzUa
voltage_array = []
mean_TzUa = []
with open('TzUa_data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        voltage_array.append(float(row[0]))
        mean_TzUa.append(float(row[1]))
slope, intercept, _, _, _ = stats.linregress(voltage_array, [x * 1000 for x in mean_TzUa])
regression_line = slope * np.array(voltage_array) + intercept

axs[1, 3].scatter(voltage_array, [x * 1000 for x in mean_TzUa], marker='o', s=100, label='$yaw\ torque$', color='lightgreen')
axs[1, 3].plot(voltage_array, regression_line, color='red')
axs[1, 3].set_xticks(np.arange(5, 15.5, 1))
axs[1, 3].tick_params(axis='x', labelsize=15)
axs[1, 3].set_yticks(np.arange(-0.5, 0.51, 0.1))
axs[1, 3].tick_params(axis='y', labelsize=15)
axs[1, 3].set_xlabel('$U_a$(V)', fontsize=20)
axs[1, 3].set_ylabel('yaw torque (N $\\cdot$ mm)', fontsize=20)
xlim = axs[1, 3].get_xlim()
ylim = axs[1, 3].get_ylim()
x_pos = xlim[1]
y_pos = 0.95*ylim[0]
axs[1, 3].text(x_pos, y_pos, r'$U_r=0$'+'\n'+r'$U_p=0$'+'\n'+r'$\Delta \sigma=0.1$', fontsize=cornersize,ha='right')
axs[1, 3].legend(fontsize=legendsize, loc='upper left')

plt.tight_layout()  # 调整子图参数以给定指定的填充
fig.subplots_adjust(left=0.07, right=0.99,bottom=0.16, wspace=0.43,hspace=0.42)
plt.show()

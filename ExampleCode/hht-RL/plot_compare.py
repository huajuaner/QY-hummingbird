'''Plotting the curve of reward changes over steps, comparing different Algo.'''
import numpy as np
import matplotlib.pyplot as plt

#PPO 1
#A2C 2
#DDPG 3
#SAC 4

# DEFAULT_OUTPUT_FOLDER = 'hover_results'
DEFAULT_OUTPUT_FOLDER = 'att_results'
path1 = DEFAULT_OUTPUT_FOLDER+'/save-09.05.2024_19.42.15/evaluations.npz'

# 加载评估结果
evaluations = np.load(path1)
values = evaluations.values()

# Extract reward and timestep data from the evaluation
# 5e7 totol steps, 5e7/2e4 evaluations timesteps
rewards = np.array(evaluations['results'])[-1250:,:]/1e4
rewards_c = np.clip(rewards, -2, 10)
steps = (np.array(evaluations['timesteps'])[-1250:]-8440000)*2/1e7
lengths = evaluations['ep_lengths']

start_index =  0 
end_index = steps.shape[0] 

print(steps.shape)

for i in range(1,5):
    plt.plot(steps[start_index:end_index], rewards_c[start_index:end_index,i], color=(0.7, 0.8, 0.9),linewidth=0.5)

# Calculate the mean every stride data points
stride=10
mean_rewards = np.mean(rewards[start_index:end_index-1:stride], axis=1)
new_steps = np.arange(steps[start_index], steps[-1], stride*4e4/1e7)
# Plot the mean curve
plt.plot(new_steps, np.clip(mean_rewards,-2,10), color='tab:blue', label = 'PPO',)


################################
#
#sac
#
####################################
path1 = DEFAULT_OUTPUT_FOLDER+'/save-11.30.2024_22.31.53/evaluations.npz'

# 加载评估结果
evaluations = np.load(path1)
values = evaluations.values()

# Extract reward and timestep data from the evaluation
# 5e7 totol steps, 5e7/2e4 = 2500 evaluations timesteps
# rewards = np.array(evaluations['results'])[0:1250,:]/1e5
# rewards_c = np.clip(rewards, -2, 10)
# steps = (np.array(evaluations['timesteps'])[0:1250])*2/1e7
rewards = np.array(evaluations['results'])[0:1250,:]/5e4
rewards_c = np.clip(rewards, -2, 10)
steps = (np.array(evaluations['timesteps'])[0:1250])*2/1e7
lengths = evaluations['ep_lengths']

start_index =  0 
end_index = steps.shape[0] 

print('SAC')
print(rewards.shape)
print(steps.shape)

for i in range(1,5):
    plt.plot(steps[start_index:end_index], rewards_c[start_index:end_index,i], color=(0.8, 0.9, 0.8),linewidth=0.2)

# Calculate the mean every stride data points
stride=10
mean_rewards = np.mean(rewards[start_index:end_index-1:stride], axis=1)
new_steps = np.arange(steps[start_index], steps[-1], stride*4e4/1e7)
# Plot the mean curve
plt.plot(new_steps, np.clip(mean_rewards,-2,10), color='tab:green', label = 'SAC',)






###########

#A2C

#######################
path2 = DEFAULT_OUTPUT_FOLDER+'/save-09.06.2024_17.03.48/evaluations.npz'

# 加载评估结果
evaluations = np.load(path2)
values = evaluations.values()

# Extract reward and timestep data from the evaluation
# 5e7 totol steps, 5e7/2e4 evaluations timesteps
rewards = np.array(evaluations['results'][0:1250,:])/1e4
rewards_c = np.clip(rewards,-2,10)
steps = (np.array(evaluations['timesteps'][0:1250]))*2/1e7
lengths = evaluations['ep_lengths']

start_index =  0 
end_index = steps.shape[0] 

print(steps.shape)

for i in range(1,5):
    plt.plot(steps[start_index:end_index], rewards_c[start_index:end_index,i], color=(1, 0.8, 0.6),linewidth=0.2)

# Calculate the mean every stride data points
stride=10
mean_rewards = np.mean(rewards[start_index:end_index-1:stride], axis=1)
new_steps = np.arange(steps[start_index], steps[-1], stride*4e4/1e7)
# Plot the mean curve
plt.plot(new_steps, mean_rewards, color='tab:orange',  label = 'A2C',)


path3 = DEFAULT_OUTPUT_FOLDER+'/save-09.08.2024_09.05.44/evaluations.npz'

# 加载评估结果
evaluations = np.load(path3)
values = evaluations.values()

# Extract reward and timestep data from the evaluation
# 5e7 totol steps, 5e7/2e4 evaluations timesteps
rewards = np.array(evaluations['results'][0:1250,:])/1e4
rewards_c = np.clip(rewards, -2,10)
steps = np.array(evaluations['timesteps'][0:1250])*2/1e7
lengths = evaluations['ep_lengths']

start_index =  0 
end_index = steps.shape[0] 

print(steps.shape)

for i in range(1,5):
    plt.plot(steps[start_index:end_index], rewards_c[start_index:end_index,i], color=(1, 0.6, 0.6),linewidth=0.2)

# Calculate the mean every stride data points
stride=10
mean_rewards = np.mean(rewards[start_index:end_index-1:stride], axis=1)
new_steps = np.arange(steps[start_index], steps[-1], stride*4e4/1e7)
# Plot the mean curve
plt.plot(new_steps, mean_rewards, color='tab:red', label = 'DDPG')





##################

#plot_set

####################
tick_size =20
plt.gca().set_xticks([0,1,2,3,4,5])
plt.gca().set_yticks([-2,-1,0,1,2,3,4,5,6,7,8,9,10])
# plt.gca().set_xticks([0,1e7,2e7,3e7,4e7,5e7])
# plt.gca().set_yticks([0,1e5,2e5,3e5,4e5,5e5,6e5])
plt.tick_params(axis='x', labelsize=tick_size )
plt.gca().xaxis.offsetText.set_fontsize(tick_size-3)
plt.gca().yaxis.offsetText.set_fontsize(tick_size-3)
plt.tick_params(axis='y', labelsize=tick_size )
plt.xlabel('Timesteps ($10^7$)',fontsize=tick_size)
plt.ylabel('Episode Reward ($10^4$)',fontsize=tick_size)
# plt.title('Reward over Timesteps',fontsize=tick_size-2)
plt.grid(True)
plt.legend(fontsize=14)
plt.show()
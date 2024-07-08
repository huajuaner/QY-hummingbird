'''Plotting the curve of reward changes over steps.'''
import numpy as np
import matplotlib.pyplot as plt

# DEFAULT_OUTPUT_FOLDER = 'hover_results'
DEFAULT_OUTPUT_FOLDER = 'att_results'
path = DEFAULT_OUTPUT_FOLDER+'/save-07.04.2024_17.39.03/evaluations.npz'

# Load evaluation results
evaluations = np.load(path)
values = evaluations.values()


print("Values of all keys:", values)
# Extract reward and timestep data from the evaluation
# 5e7 totol steps, 5e7/2e4 evaluations timesteps
rewards = np.array(evaluations['results'])/1e5
steps = np.array(evaluations['timesteps'])/1e7
lengths = evaluations['ep_lengths']

start_index =  0 
end_index = steps.shape[0] 

print(steps.shape)

for i in range(1,5):
    plt.plot(steps[start_index:end_index], rewards[start_index:end_index,i], color=(0.7, 0.8, 0.9),linewidth=0.5)

# Calculate the mean every stride data points
stride=10
mean_rewards = np.mean(rewards[start_index:end_index:stride], axis=1)
new_steps = np.arange(steps[start_index], steps[-1], stride*2e4/1e7)
# Plot the mean curve
plt.plot(new_steps, mean_rewards, color='tab:blue')
# label='Mean Reward of 5 Episodes'

tick_size =20
plt.gca().set_xticks([0,1,2,3,4,5])
# plt.gca().set_xticks([0,1e7,2e7,3e7,4e7,5e7])
# plt.gca().set_yticks([0,1e5,2e5,3e5,4e5,5e5,6e5])
plt.tick_params(axis='x', labelsize=tick_size )
plt.gca().xaxis.offsetText.set_fontsize(tick_size-3)
plt.gca().yaxis.offsetText.set_fontsize(tick_size-3)
plt.tick_params(axis='y', labelsize=tick_size )
plt.xlabel('Timesteps ($10^7$)',fontsize=tick_size)
plt.ylabel('Episode Reward ($10^5$)',fontsize=tick_size)
# plt.title('Reward over Timesteps',fontsize=tick_size-2)
plt.grid(True)
plt.legend()
plt.show()

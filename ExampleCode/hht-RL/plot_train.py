import numpy as np
import matplotlib.pyplot as plt

# DEFAULT_OUTPUT_FOLDER = 'hover_results'
DEFAULT_OUTPUT_FOLDER = 'att_results'
path = DEFAULT_OUTPUT_FOLDER+'/save-04.25.2024_00.22.38/evaluations.npz'

# 加载评估结果
evaluations = np.load(path)

# 提取评估中的奖励和时间步数数据
rewards = evaluations['results']
steps = evaluations['timesteps']

# 绘制奖励随时间步数变化的曲线
plt.plot(steps, rewards)
# plt.plot(steps, rewards[:,0])
plt.xlabel('Timesteps')
plt.ylabel('Episode Reward')
plt.title('Reward over Timesteps')
plt.grid(True)
plt.show()
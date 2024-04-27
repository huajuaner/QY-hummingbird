import numpy as np

DEFAULT_OUTPUT_FOLDER = 'results'
path = DEFAULT_OUTPUT_FOLDER+'/save-04.06.2024_22.52.47/evaluations.npz'

# 读取npz文件
data = np.load(path)

# 查看文件中的所有数组名
print(data.files)
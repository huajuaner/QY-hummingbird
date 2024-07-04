'''Plotting the curve of reward changes over steps.'''
import numpy as np

DEFAULT_OUTPUT_FOLDER = 'results'
path = DEFAULT_OUTPUT_FOLDER+'/save-04.06.2024_22.52.47/evaluations.npz'

data = np.load(path)
print(data.files)
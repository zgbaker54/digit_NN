
import numpy as np
import matplotlib.pyplot as plt

from NN import *
from nn_tools import *

# load data
data = get_data('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
ims = data['ims']
labs = data['labs']

x = NN(log_file='values_2ndFullRun_alpha1e-1_4Levels_2020-02-05_04_51.pckl')

misses = 0

runs = 10000
for i in range(runs):
    submit = np.expand_dims(data['ims'][i,:], 1) / 255
    expected = label_2_array(data['labs'][i])
    x.propogate(submit)
    actual = int(data['labs'][i])
    guess = int(np.argmax(x.get_output()))
    
    print('Run: ' + str(i) + ' | Actual/Guess: ' + str(actual) + '/' + str(guess))
    
    if guess != actual:
        misses += 1

plt.imshow(np.reshape(data['ims'][i], [28,28]))

print(100 * misses/runs)
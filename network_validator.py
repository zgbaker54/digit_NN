
import numpy as np
import matplotlib.pyplot as plt
from NN import *
from nn_tools import *

# run this code to determine the error rate of the chosen NN

# select NN
#x = NN(log_file='values_fullRun_alpha1e-2_2020-02-03_08_07.pckl')             # ER: 56.15%
#x = NN(log_file='values_fullRun_alpha1e-1_4Levels_2020-02-04_07_30.pckl')     # ER: 7.45%
#x = NN(log_file='values_2ndFullRun_alpha1e-1_4Levels_2020-02-05_04_51.pckl')  # ER: 5.31%
#x = NN(log_file='values_3rdFullRun_alpha1e-1_4Levels_2020-02-06_15_44.pckl')  # ER: 4.52%
x = NN(log_file='values_4thFullRun_alpha1e-1_4Levels_2020-02-08_10_05.pckl')   # ER: 4.14%

# load data
data = get_data('t10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
ims = data['ims']
labs = data['labs']

# run validation
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

# print error rate (%)
print(100 * misses/runs)

import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle

from NN import *
from nn_tools import *

# load data
data = get_data('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
ims = data['ims']
labs = data['labs']

#x = NN(shape=[784,300,100,10], alpha=0.1, beta=0.3)
#x = NN(log_file='values_fullRun_alpha1e-1_4Levels_2020-02-04_07_30.pckl')
#x = NN(log_file='values_2ndFullRun_alpha1e-1_4Levels_2020-02-05_04_51.pckl')
x = NN(log_file='values_3rdFullRun_alpha1e-1_4Levels_2020-02-06_15_44.pckl')

before = datetime.datetime.now()
runs = 60000
for i in range(runs):
    submit = np.expand_dims(ims[i,:], 1) / 255
    expected = label_2_array(labs[i])
    x.train(submit, expected)
    actual = int(labs[i])
    guess = int(np.argmax(x.get_output()))
    print('Run: ' + str(i) + ' | Actual/Guess: ' + str(actual) + '/' + str(guess) + ' | ' + str(x.costLog['sums'][-1]))
after = datetime.datetime.now()
print(after-before)

plt.plot(x.costLog['sums'])

# save network
fname = str(after).split('.')
fname = fname[0] + '_' + fname[1]
fname = fname.split(' ')
fname = fname[0] + '_' + fname[1]
fname = fname.split(':')
fname = fname[0] + '_' + fname[1]
fname = os.path.join(os.getcwd(), 'logs', 'values_' + fname + '.pckl')

with open(fname, 'wb') as fid:
    pickle.dump(x, fid)
#
## trained_n = NN(log_file='values_fullRun_alpha1e-1_4Levels_2020-02-04_07_30.pckl')


import os
import numpy as np
import matplotlib.pyplot as plt
import datetime
import pickle
from NN import *
from nn_tools import *

# train network

# initialize a new NN or load a previous one from a .pckl file in the logs dir
 x = NN(shape=[784,300,100,10], alpha=0.1, beta=0)                             # new NN
# x = NN(log_file='values_3rdFullRun_alpha1e-1_4Levels_2020-02-06_15_44.pckl') # loaded NN

# load data
data = get_data('train-images-idx3-ubyte', 'train-labels-idx1-ubyte')
ims = data['ims']
labs = data['labs']

# run training
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

# plot cost of each training after propogation
plt.plot(x.costLog['sums'])

# get file name for network
fname = str(after).split('.')
fname = fname[0] + '_' + fname[1]
fname = fname.split(' ')
fname = fname[0] + '_' + fname[1]
fname = fname.split(':')
fname = fname[0] + '_' + fname[1]
fname = os.path.join(os.getcwd(), 'logs', 'values_' + fname + '.pckl')

# save network
with open(fname, 'wb') as fid:
    pickle.dump(x, fid)


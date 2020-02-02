
import os
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import random
import datetime
import pickle

# set seed for consistency
np.random.seed(12345)

# load data
dataDir = os.path.join(os.getcwd(), 'data')
dataFiles = os.listdir(dataDir)
with open(os.path.join(dataDir, 'train-images-idx3-ubyte'), 'rb') as fid:
    ims_proper = idx2numpy.convert_from_file(fid)
ims_proper = ims_proper.astype('float')
with open(os.path.join(dataDir, 'train-labels-idx1-ubyte'), 'rb') as fid:
    labs = idx2numpy.convert_from_file(fid)
labs = labs.astype('float')
ims = np.squeeze(np.reshape(ims_proper, [ims_proper.shape[0], -1, 1]))

# create class to generate neural network
class NN:
    
    def __init__(self, shape, alpha=0.01):
        
        self.alpha = alpha
        self.shape = shape
        self.n_levels = len(shape)
        self.zero_az()
        self.init_wb()
        self.zero_deltas()
        
        self.costLog = {}
        self.costLog['vals'] = []
        self.costLog['sums'] = []
    
    def zero_az(self):
        self.a = []
        for i in range(self.n_levels):
            self.a.append(np.zeros([self.shape[i],1], dtype='float'))
        self.z = []
        for i in range(1, self.n_levels):
            self.z.append(np.zeros([self.shape[i],1], dtype='float'))
    
    def init_wb(self):
        self.w = []
        for i in range(1, self.n_levels):
            rands = np.random.rand(self.shape[i], self.shape[i-1]) * 2 - 1
            self.w.append(rands)
        self.b = []
        for i in range(1, self.n_levels):
            rands = np.random.rand(self.shape[i], 1) * 2 - 1
            self.b.append(rands)
        
    def zero_deltas(self):
        self.da = []
        for i in range(self.n_levels):
            self.da.append(np.zeros([self.shape[i],1], dtype='float'))
        self.dz = []
        for i in range(self.n_levels):
            self.dz.append(np.zeros([self.shape[i],1], dtype='float'))
        self.dw = []
        for i in range(1,self.n_levels):
            self.dw.append(np.zeros([self.shape[i], self.shape[i-1]], dtype='float'))
        self.db = []
        for i in range(1,self.n_levels):
            self.db.append(np.zeros([self.shape[i],1], dtype='float'))
    
    
    def train(self, submit, expected):
        
        # add submit data to Level 0 Node inputs
        if submit.shape != (784,1):
            raise Exception("Submitted data must match the shape of the network's self.a[0]")
        
        # propogate submission
        self.zero_az() # may not need
        self.a[0] = submit
        for i in range(self.n_levels-1):
            self.z[i] = np.matmul(self.w[i], self.a[i]) + self.b[i]
            self.a[i+1] = sigmoid(self.z[i])
        
        # record results
        result = self.a[self.n_levels - 1]
        cost = (1/2) * (expected - result) ** 2
        delta_cost = expected - result
        self.costLog['vals'].append(cost)
        self.costLog['sums'].append(sum(cost))
        
        
        # backpropogation
        self.zero_deltas() # may not need
        self.da[self.n_levels - 1] = delta_cost
        for L in reversed(range(self.n_levels - 1)):
            # dz
            self.dz[L] = delta_sigmoid(self.z[L]) * self.da[L + 1]
            
            # db
            self.db[L] = self.dz[L] * 1
            
            # dw
            for k in range(self.w[L].shape[1]):
                self.dw[L][:,k] = self.a[L][k]
                self.dw[L][:,k] *= np.squeeze(self.dz[L])
            
            # da
            for j in range(self.w[L].shape[0]):
                for k in range(self.w[L].shape[1]):
                    self.da[L][k] += self.w[L][j,k] * self.dz[L][j]

        # apply deltas to weights and biases
        for L in range(self.n_levels - 1):
            self.w[L] += self.alpha * self.dw[L]
            self.b[L] += self.alpha * self.db[L]

    def get_output(self):        
        return self.a[self.n_levels - 1]
        
    def __str__(self):
        return 'NN Obj: ' + str(self.shape)

    def __repr__(self):
        return str(self)

def label_2_array(lab):
    result = np.zeros([10], dtype = 'float')
    result[int(lab)] = 1.0
    result = np.expand_dims(result, 1)
    return result
    
def array_2_label(array):
    error = (array - 1) ** 2
    return np.argmin(error)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def delta_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


x = NN([784,300,10], alpha=0.1)

#runs = 1
#for i in range(runs):
#    submit = np.expand_dims(ims[i,:], 1)
#    expected = label_2_array(labs[i])
#    x.train(submit, expected)

before = datetime.datetime.now()

#s1 = plt.subplot(3,1,1)
#s2 = plt.subplot(3,1,2)
#s3 = plt.subplot(3,1,3)

runs = 60000
for i in range(runs):
    submit = np.expand_dims(ims[i,:], 1) / 255
    expected = label_2_array(labs[i])
    x.train(submit, expected)
    actual = int(labs[i])
    guess = int(np.argmax(x.get_output()))
    print('Run: ' + str(i) + ' | Actual/Guess: ' + str(actual) + '/' + str(guess) + ' | ' + str(x.costLog['sums'][-1]))
    
#    s1.plot(x.a[0])
#    s2.plot(x.a[1])
#    s3.plot(x.a[2])
    

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

# save code
with open(fname, 'wb') as fid:
    pickle.dump(x, fid)

# load code
#with open(os.path.join(os.getcwd(), 'logs', 'filename.pckl'), 'rb') as fid:
#    x = pickle.load(fid)
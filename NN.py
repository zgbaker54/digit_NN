
import os
import numpy as np
import pickle

# create class to generate neural network
class NN:
    
    def __init__(self, shape=[784,300,100,10], alpha=0.01, log_file=None):
        
        if log_file == None:
            self.alpha = alpha
            self.shape = shape
            self.n_levels = len(shape)
            self.zero_az()
            self.init_wb()
            self.zero_deltas()
            
            self.costLog = {}
            self.costLog['vals'] = []
            self.costLog['sums'] = []
            
        else:
            print('NN will be loaded from ' + log_file + '; ignoring other arguments...')
            
            # load code
            with open(os.path.join(os.getcwd(), 'logs', log_file), 'rb') as fid:
                x = pickle.load(fid)
            self.alpha = x.alpha
            self.shape = x.shape
            self.n_levels = x.n_levels
            self.a = x.a
            self.z = x.z
            self.w = x.w
            self.b = x.b
            self.da = x.da
            self.dz = x.dz
            self.dw = x.dw
            self.db = x.db
            self.costLog = x.costLog
            
            
    
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
        
        # propogate submission
        self.propogate(submit)
        
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
            self.dz[L] = self.delta_sigmoid(self.z[L]) * self.da[L + 1]
            
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
            
    def propogate(self, submit):
        
        # ensure submit is properly shaped
        if submit.shape != (784,1): # fix this so it isn't hard coded
            raise Exception("Submitted data must match the shape of the network's self.a[0]")
        
        self.zero_az() # may not need
        self.a[0] = submit
        for i in range(self.n_levels-1):
            self.z[i] = np.matmul(self.w[i], self.a[i]) + self.b[i]
            self.a[i+1] = self.sigmoid(self.z[i])
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def delta_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def get_output(self):        
        return self.a[self.n_levels - 1]
        
    def __str__(self):
        return 'NN Obj: ' + str(self.shape)

    def __repr__(self):
        return str(self)
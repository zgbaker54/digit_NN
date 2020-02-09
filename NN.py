
import os
import numpy as np
import pickle

# create class to generate neural network
class NN:
    
    # initialize NN
    def __init__(self, shape=[784,300,100,10], alpha=0.01, beta=0.0, log_file=None):
        
        # initialize a new NN with specified arguments if no log_file is given
        if log_file == None:
            self.alpha = alpha                # learning rate
            self.beta = beta                  # momentum
            self.shape = shape                # shape of NN
            self.n_levels = len(shape)        # number of levels in NN
            self.zero_az()
            self.init_wb()
            self.zero_deltas()
            self.costLog = {}                 # keeps track of cost after each training example
            self.costLog['vals'] = []
            self.costLog['sums'] = []
        
        # load the NN from the .pckl file if log_file is given (ignore all other arguments)
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
            if hasattr(x, 'beta'):
                self.beta = x.beta
            else:
                self.beta = 0
    
    
    # reset (or initialize) arrays of zeros for activations (self.a) and summations(self.z) of each node 
    def zero_az(self):
        
        self.a = []
        for i in range(self.n_levels):
            self.a.append(np.zeros([self.shape[i],1], dtype='float'))
        self.z = []
        for i in range(1, self.n_levels):
            self.z.append(np.zeros([self.shape[i],1], dtype='float'))
    
    
    # initialize the weights (self.w) and biases (self.b) to be random values from [-1 1]
    def init_wb(self): 
        
        # set seed for consistency
        np.random.seed(22345)
        
        # weights
        self.w = []
        for i in range(1, self.n_levels):
            rands = np.random.rand(self.shape[i], self.shape[i-1]) * 2 - 1
            self.w.append(rands)
        
        # biases
        self.b = []
        for i in range(1, self.n_levels):
            rands = np.random.rand(self.shape[i], 1) * 2 - 1
            self.b.append(rands)
    
    
    # reset (or initialize) arrays of zeros for the gradient of a, z, w, and b
    def zero_deltas(self):
        
        # da
        self.zero_da()
        
        # dz
        self.dz = []
        for i in range(self.n_levels):
            self.dz.append(np.zeros([self.shape[i],1], dtype='float'))
        
        # dw
        self.dw = []
        for i in range(1,self.n_levels):
            self.dw.append(np.zeros([self.shape[i], self.shape[i-1]], dtype='float'))
        
        # db
        self.db = []
        for i in range(1,self.n_levels):
            self.db.append(np.zeros([self.shape[i],1], dtype='float'))
    
    
    # reset (or initialize) arrays of zeros for the gradient of a only
    def zero_da(self):
        self.da = []
        for i in range(self.n_levels):
            self.da.append(np.zeros([self.shape[i],1], dtype='float'))
    
    
    # propogate and backpropogate the NN with training example submit, based on expected
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
        self.da[self.n_levels - 1] = delta_cost
        for L in reversed(range(self.n_levels - 1)):
            
            # dz
            self.dz[L] = self.delta_sigmoid(self.z[L]) * self.da[L + 1]
            
            # db
            self.db[L] = self.dz[L] * 1 + self.db[L] * self.beta
            
            # dw
            for k in range(self.w[L].shape[1]):
                self.dw[L][:,k] = self.a[L][k] * np.squeeze(self.dz[L]) + self.dw[L][:,k] * self.beta
            
            # da
            self.zero_da()
            for j in range(self.w[L].shape[0]):
                for k in range(self.w[L].shape[1]):
                    self.da[L][k] += self.w[L][j,k] * self.dz[L][j]
        
        # apply deltas to weights and biases
        for L in range(self.n_levels - 1):
            self.w[L] += self.alpha * self.dw[L]
            self.b[L] += self.alpha * self.db[L]
    
    
    # input submit to the first level and iterate through all levels, solving for z and a values
    def propogate(self, submit):
        
        # ensure submit is properly shaped
        if submit.shape != (self.shape[0], 1):
            raise Exception("Submitted data must match the shape of the network's self.a[0]")
        
        # input submit to first level
        self.a[0] = submit
        
        # solve for subsequent levels' z and a values based on w and b
        for i in range(self.n_levels-1):
            self.z[i] = np.matmul(self.w[i], self.a[i]) + self.b[i]
            self.a[i+1] = self.sigmoid(self.z[i])
    
    
    # the activation function of each node
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    
    # the derivative of the activation function
    def delta_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    
    # query the NN for its current output from propogation
    def get_output(self):        
        return self.a[self.n_levels - 1]
    
    
    # adjust str output
    def __str__(self):
        return 'NN Obj: ' + str(self.shape)


    # adjust repr
    def __repr__(self):
        return str(self)


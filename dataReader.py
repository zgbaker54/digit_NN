
import os
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import random
import datetime
import pickle

random.seed('needaseed')

data = {}

dataDir = os.path.join(os.getcwd(), 'data')
dataFiles = os.listdir(dataDir)
#print(dataFiles)

with open(os.path.join(dataDir, 'train-images-idx3-ubyte'), 'rb') as fid:
    ims_proper = idx2numpy.convert_from_file(fid)
ims_proper = ims_proper.astype('float')

with open(os.path.join(dataDir, 'train-labels-idx1-ubyte'), 'rb') as fid:
    labs = idx2numpy.convert_from_file(fid)
labs = labs.astype('float')

#plt.imshow(ims_proper[0,:,:], cmap='gray')

ims = np.squeeze(np.reshape(ims_proper, [ims_proper.shape[0], -1, 1]))


class NN:
    
    def __init__(self, shape, alpha = 0.01):
        
        self.shape = shape
        self.alpha = alpha
        self.network = []
        self.n_levels = len(shape)
        self.costLog = {}
        self.costLog['vals'] = []
        self.costLog['sum'] = []
        
        # Create Nodes
        for level,dim in enumerate(self.shape):
            self.network.append([])
            for i in range(dim):
                self.network[level].append(Node(level, bias = random.random()))
        
        # Connect Nodes
        for level,layer in enumerate(self.network):
            if level == 0:
                continue
            
            prevLayer = self.network[level - 1]
            
            for node in layer:
                for prevLayerNode in prevLayer:
                    node.inputWeights.append(Weight(prevLayerNode, node, a = random.random()))
    
    def train(self, submit, expected):
        
        # refresh all Node inputs
        for layer in self.network:
            for node in layer:
                node.refresh_input()
        
        # add submit data to Level 0 Node inputs
        if submit.shape != (784,):
            raise Exception("Submitted data must match the shape of the network's Level 0")
        for i,inputNode in enumerate(self.network[0]):
            inputNode.input = submit[i]
        
        # for each Node, starting at Level 0...       
        for layer in self.network:
            for node in layer:
                # - add values from all input weights and bias to node's input
                node.add_weight_values()
                node.add_bias()
                
                # - initiate sigmoid function to update output
                node.execute_sigmoid()
        
        # find difference between network output and expected
        cost = (expected - self.get_output()) ** 2
        self.costLog['vals'].append(cost)
        self.costLog['sum'].append(sum(cost))
#        delta_cost = 2 * (self.get_output() - expected)
        delta_cost = 2 * (expected - self.get_output())
        
        
        ######## backpropogate NN #############
        
        # node.input is z
        # node.output is a
        
        # refresh all deltas
        for layer in self.network:
            for node in layer:
                node.refresh_deltas()
        
        # calculate deltas
        for idx,layer in enumerate(reversed(self.network)):
            level = len(self.network) - idx - 1
            for node in layer:
                # behave different for last level
                if level == self.n_levels - 1:
                    node.dcda = sum(delta_cost)
                else:
                    
                    laterLayer = self.network[level + 1]
                    for laterNode in laterLayer:
                        node.dcda += laterNode.dcda # you have to do the sum of all dcda*dadz*dzda
#                        # dzda is the weight of the Weight going in
#                    for laterNode in laterLayer:
#                        for weight in node.inputWeights:
#                            node.dcda += weight.a * node.dadz * laterNode.dcda
                        
                    node.dadz = delta_sigmoid(node.output)
                    node.dzdb = 1
                    node.dcdb = node.dcda * node.dadz * node.dzdb
                    
                for weight in node.inputWeights:
                    weight.dzdw = weight.inputNode.output
                    weight.dcdw = node.dcda * node.dadz * weight.dzdw
        
        # apply deltas with alpha
        for layer in self.network:
            for node in layer:
                node.bias += self.alpha * node.dcdb
                for weight in node.inputWeights:
                    weight.a += self.alpha * weight.dcdw
                
            
                
    def get_output(self):
        last_level = len(self.shape) - 1
        last_layer = self.network[last_level]
        
        output = np.zeros([self.shape[last_level]], dtype = 'float')
        
        for i,node in enumerate(last_layer):
            output[i] = node.output
        
        return output
    
    def __str__(self):
        return 'NN Obj: ' + str(self.shape)

    def __repr__(self):
        return str(self)
            
            

class Node:
    
    ctr = 0
    
    def __init__(self, level, bias = 0.0):   
        
        Node.ctr += 1
        
        self.level = level
        self.bias = bias
        self.inputWeights = []
        self.input = 0.0
        self.output = 0.0
        self.id = Node.ctr
        self.dcda = 0.0
        self.dadz = 0.0
        self.dzdb = 0.0
        self.dcdb = 0.0
    
    def refresh_input(self):
        self.input = 0.0
    
    def add_weight_values(self):
        for weight in self.inputWeights:
            self.input += weight.getOut()
    
    def add_bias(self):
        self.input += self.bias
    
    def execute_sigmoid(self):
        self.output = sigmoid(self.input)
    
    def refresh_deltas(self):
        self.dcda = 0.0
        self.dadz = 0.0
        self.dzdb = 0.0
        self.dcdb = 0.0
        for weight in self.inputWeights:
            weight.refresh_deltas()

    
    def __str__(self):
        return 'Node Obj: L-' + str(self.level) + ' id-' + str(self.id)
    
    def __repr__(self):
        return str(self)

class Weight:
    
    ctr = 0
    
    def __init__(self, inputNode, outputNode, a = 0.0):
        
        Weight.ctr += 1
        
        self.inputNode = inputNode
        self.outputNode = outputNode
        self.a = a
        self.id = Weight.ctr
        self.dcdw = 0.0
        self.dzdw = 0.0
    
    def getOut(self):
        return self.inputNode.output * self.a
    
    def refresh_deltas(self):
        self.dcdw = 0.0
        self.dzdw = 0.0
    
    def __str__(self):
        result = 'Weight Obj: L-' + str(self.inputNode.level)
        result += ',' + str(self.outputNode.level)
        result += ' id-' + str(self.id)
        return result
    
    def __repr__(self):
        return str(self)
        

# converts number n ranging from 1-9 to an array of zeros but with the nth
# index as one 
def label_2_array(lab):
    result = np.zeros([10], dtype = 'float')
    result[int(lab)] = 1.0
    return result
    
def array_2_label(array):
    error = (array - 1) ** 2
    return np.argmin(error)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def delta_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
        

x = NN([784,300,10])
#x = NN([5,4,3,2])

runs = 30

before = datetime.datetime.now()

for i in range(runs):
    submit = ims[i,:]
    expected = label_2_array(labs[i])
    x.train(submit, expected)
    
    actual = int(labs[i])
    guess = int(np.argmax(x.get_output()))
    
    print('Run: ' + str(i) + ' | Actual/Guess: ' + str(actual) + '/' + str(guess) + ' | ' + str(x.costLog['sum'][-1]))

after = datetime.datetime.now()

print(after-before)

plt.plot(x.costLog['sum'])

fname = str(after).split('.')
fname = fname[0] + '_' + fname[1]
fname = fname.split(' ')
fname = fname[0] + '_' + fname[1]
fname = fname.split(':')
fname = fname[0] + '_' + fname[1]

fname = os.path.join(os.getcwd(), 'logs', 'values_' + fname + '.pckl')

f = open(fname, 'wb')
pickle.dump(x, f)
f.close()



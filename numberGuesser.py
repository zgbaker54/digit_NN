import numpy as np
from uiFigureClass import *
from NN import *
from nn_tools import *
import scipy.interpolate
import matplotlib.pyplot as plt

network = NN(log_file='values_3rdFullRun_alpha1e-1_4Levels_2020-02-06_15_44.pckl')

myFig = UIFigure()
print('Draw a nuumber and then close the window...')
myFig.run()

drawing = np.mean(np.array(myFig.im, dtype='float'), 2)

if np.sum(drawing) == 0:
    raise(Exception('No drawing detected'))

# center drawing
x_sum = 0
y_sum = 0
for i in range(drawing.shape[0]):
    for j in range(drawing.shape[1]):
        x_sum += drawing[i, j] * j
        y_sum += drawing[i, j] * i
x_cent = x_sum / np.sum(drawing)
y_cent = y_sum / np.sum(drawing)

x_shift = int(np.round(drawing.shape[1]/2 - x_cent))
y_shift = int(np.round(drawing.shape[0]/2 - y_cent))

drawing = np.roll(drawing, x_shift, axis=1)
drawing = np.roll(drawing, y_shift, axis=0)


# trim drawing outer space
while drawing.shape != (1,1):
    if all(drawing[:, 0] == 0) and all(drawing[0, :] == 0) and all(drawing[-1, :] == 0) and all(drawing[:, -1] == 0):
        drawing = drawing[1:-1,1:-1]
    else:
        break

# interpolate to 20x20
x = np.arange(0, drawing.shape[1], 1)
y = np.arange(0, drawing.shape[0], 1)

interp_f = scipy.interpolate.interp2d(x, y, drawing)

xq = np.linspace(0, drawing.shape[1], 20)
yq = np.linspace(0, drawing.shape[0], 20)
drawing = interp_f(xq, yq)

formed_drawing = np.zeros([28, 28])
formed_drawing[5:25, 5:25] = drawing

#plt.imshow(formed_drawing)


submit = np.expand_dims(np.squeeze(np.reshape(formed_drawing, [-1, 1])), 1)
network.propogate(submit)
print(np.argmax(network.get_output()))
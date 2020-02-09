
import os
import numpy as np
import idx2numpy

# supporting tools for network_trainer.py and network_validator.py

# returns data from idx file as a dictionary of np arrays
def get_data(data_file, label_file):
    dataDir = os.path.join(os.getcwd(), 'data')
    with open(os.path.join(dataDir, data_file), 'rb') as fid:
        ims_proper = idx2numpy.convert_from_file(fid)
    ims_proper = ims_proper.astype('float')
    with open(os.path.join(dataDir, label_file), 'rb') as fid:
        labs = idx2numpy.convert_from_file(fid)
    labs = labs.astype('float')
    ims = np.squeeze(np.reshape(ims_proper, [ims_proper.shape[0], -1, 1]))
    data = {}
    data['ims'] = ims
    data['labs'] = labs
    return data

# convert label number into array of zeros with a 1 at the index of the label
# eg 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
def label_2_array(lab):
    result = np.zeros([10], dtype = 'float')
    result[int(lab)] = 1.0
    result = np.expand_dims(result, 1)
    return result

# convert array into the most appropriate label
# eg [0, 0.001, 0, 1, 0, 0.4, 0, 0, 0, 0] -> 3
def array_2_label(array):
    error = (array - 1) ** 2
    return np.argmin(error)
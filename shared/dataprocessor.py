
import h5py
import numpy as np
import os

def h5tonpy(h5file, outputPath, outputName, key = 'df'):
    with h5py.File(h5file, 'r') as f:
        data = f[key]['block0_values']
        np.save(os.path.join(outputPath, outputName), data)
        
        
        
def pkltonpy(pklfile, outputPath, outputName):
    import pickle
    with open(pklfile, 'rb') as f:
        data = pickle.load(f)
        data = np.array(data)
        np.save(os.path.join(outputPath, outputName), data)
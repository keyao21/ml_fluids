import os
import numpy as np
from scipy import linalg 
from scipy.sparse import rand
import matplotlib.pyplot as plt
import networkx as nx
import pdb
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime 
import logging 
import config
import util
from ESN import EchoStateNetwork

class MultiEchoStateNetwork(EchoStateNetwork):
    def __init__(self, loaddata, initLen=0, resSize=300, partial_know=True, noise=0, 
                density=0.05, spectral_radius=1.0, leaking_rate=0.0, input_scaling=1.0, ridgeReg=0, mute=False):
        """
        Major changes in this derived class: 
            - self.P is now in the __init__ function

        Dimensions:
        data : T x resSize
        Win  : resSize x inSize
        Wout : outSize x 1
        A    : resSize x resSize
        P    : outSize x resSize
        r    : 1+inSize+resSize x T
        """

        if type(loaddata)==np.ndarray:
            self.data = loaddata
        else:
            self.data = np.loadtxt(loaddata, delimiter=',')
        logging.info('Loaded data with size ', np.shape(self.data))
        if max(self.data.shape) != self.data.shape[0]: 
            logging.warning("First dimension is not the largest, make sure it is the time dimension")
            logging.warning("self.data is shape {0}".format( str(self.data.shape)))
        self.initLen = initLen
        self.trainLen = self.data.shape[0]-1
        self.inSize = self.outSize = self.data.shape[1]   # assuming same number of input and output dims
        self.resSize = resSize
        self.partial_know =  partial_know
        self.noise = noise
        self._p = density # density of reservoir network
        self.spectral_radius = spectral_radius
        self.leaking_rate = leaking_rate
        self.input_scaling = input_scaling
        self.ridgeReg = ridgeReg
        self.mute = mute # toggle for print statements
        seed = np.random.seed(42)
        self.P = np.random.rand(self.outSize, self.resSize)-0.5
        self.initialize()

    def initialize(self): 
        """This is a reimplementation of ESN object initiation to account for multiple input samples"""
        logging.info("Initializing model...")

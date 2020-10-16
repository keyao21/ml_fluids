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
from multiprocessing import Pool
import pickle 

class MultiEchoStateNetwork(EchoStateNetwork):
    def __init__(self, loaddata, initLen=0, resSize=300, partial_know=True, noise=0, 
                density=0.05, spectral_radius=1.0, leaking_rate=0.0, input_scaling=1.0, ridgeReg=0, mute=False):
        """
        Major changes in this derived class: 
            - self.P is now in the __init__ function

            - data input is now 3-dimensional, the new dimensional being the "multi" 
                aspect of this ESN implementation. There are multiple inputs threads to be converted to 
                the reservoir space to be used to regress the data in observable space to the 
                reservoir space. Note that we are dealing with time series data, but treating time as a
                dimension makes our data essentially two dimensional. the reservoir space should 
                very close to time independent, as all time information should be embedded.  

        Dimensions:
        data : T x inSize
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


        # reservoir objects
        self.Win = np.multiply( self.getRandomSparseMatrix((self.resSize, self.inSize+1), density=0.25), \
                                np.random.uniform(-1,1,(self.resSize, self.inSize+1)) )
        self.Wout = np.random.rand(self.outSize,1)-0.5
        self.A = self.getRandomGraph(self.resSize)
        self.A = np.multiply(np.random.uniform(-0.5,0.5,(self.resSize, self.resSize)), self.A)
        rhoW = max(abs(linalg.eig(self.A)[0]))
        self.A *= self.spectral_radius/rhoW

        self.P = np.random.rand(self.outSize, self.resSize)-0.5
        self.initialize()

    def convert_to_reservoir_space(self,data_chunk_idx): 
        # convert one chunk of data from observation space to reservoir space 
        u = self.data[ : self.trainLen, :, data_chunk_idx]
        v_tgt = self.data[ self.initLen+1 : self.trainLen+1, :, data_chunk_idx]

        # Sample training states (r) data
        r_chunk = np.zeros((1+self.inSize+self.resSize, self.trainLen-self.initLen+1)) 
        r_t = np.zeros((self.resSize,1))
        t_ = 0 # initialize for recording t
        for t in range(self.trainLen):
            r_t = self.leaking_rate*r_t \
                            + (1.0-self.leaking_rate)*np.tanh( np.dot(self.Win, np.vstack((1, u[t].reshape(self.inSize,1))) ) \
                                                             + np.dot( self.A, r_t) ) \
                            + self.noise*np.random.rand(self.resSize,1) #same shape as r_t
            if t >= self.initLen: 
                r_chunk[:,t_+1] = np.vstack((1,u[t].reshape(self.inSize,1), r_t)).reshape(self.inSize+self.resSize+1)
                t_ += 1

        with open('MESN_data/id{0}.pkl'.format(data_chunk_idx), 'wb') as f:
            pickle.dump(r_chunk, f)

    def initialize(self): 
        """This is a reimplementation of ESN object initiation to account for multiple input samples"""
        logging.info("Initializing model...")
        pool = Pool(processes=self.data.shape[-1])
        idxs = [i for i in range(self.data.shape[-1])]
        pool.map(self.convert_to_reservoir_space, idxs)

        r_chunks = np.zeros((1+self.inSize+self.resSize, self.trainLen-self.initLen+1, self.data.shape[-1])) 
        for data_chunk_idx in range(self.data.shape[-1]):
            with open('MESN_data/id{0}.pkl'.format(data_chunk_idx), 'rb') as f:
                r_chunk = pickle.load(f)
                r_chunks[:,:,data_chunk_idx] = r_chunk
        self.r = np.hstack([r_chunks[:,1:,i] for i in range(self.data.shape[-1])] )

        v_tgt_chunks = self.data[ self.initLen+1 : self.trainLen+1]
        self.v_tgt = np.vstack([v_tgt_chunks[:,:,i] for i in range(v_tgt_chunks.shape[-1])])


    def train(self):
        """Train on training states using linear ridge regression
        
        Find P s.t. minimizes
        ||v_d - Pr||^2 + a*||P||^2
        
        """
        logging.info("Training...")
        
        self.P = np.dot(np.dot(self.v_tgt.T,self.r.T), linalg.inv(np.dot(self.r,self.r.T) + \
            self.ridgeReg*np.eye(1+self.inSize+self.resSize) ) )


        # import pdb;pdb.set_trace()

        return



    

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

class EchoStateNetwork(object):
    def __init__(self, loaddata, initLen=0, resSize=300, partial_know=True, noise=0, 
                density=0.05, spectral_radius=1.0, leaking_rate=0.0, input_scaling=1.0, ridgeReg=0, mute=False):
        """
        Notes:
        useful - A Practical Guide to ESNs <https://pdfs.semanticscholar.org/11bb/0941b1f6088783e26d7f9603789ee1db7faa.pdf>
    

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
        self.initialize()
    
    def getRandomGraph(self, directory=None):
        """Initialize, save, or retrieve erdos-renyi graph as pickled file"""
        directory = os.path.join( config.DATA_PATH_DIR, 'graphs')
        util.set_up_dir( directory )
        filename = str(self.resSize)+'_'+str(self._p)+'.pickle'
        # Reuse random graph in the directory with the same size
        try: 
            if filename in os.listdir(directory):
                return nx.read_gpickle(directory+'/'+filename)
        except FileNotFoundError:
            logging.warning("graphs directory not found!")
        # Create new graph, save to directory 
        # calculate probability of edge creation using degree
        G = nx.fast_gnp_random_graph(self.resSize, self._p, 42, True)
        A = nx.adjacency_matrix(G).todense()      
        print( filename )
        nx.write_gpickle(A, os.path.join(directory, filename))
        return A

    def getRandomSparseMatrix(self, size, density):
        """Create sparse random matrix with 1s and 0s"""
        mat = rand(size[0], size[1], density=density)
        mat.data[:] = 1
        return mat.todense()

    def minMaxScaler(self, series):
        """Min Max scaling for 1d series"""
        scaler = MinMaxScaler()
        scaler.fit(series.reshape(-1, 1))
        scaled_series = scaler.transform(series.reshape(-1, 1))
        return scaled_series.reshape(len(scaled_series))

    def initialize(self):
        """Initilize model params and matrices"""
        logging.info("Initializing model...")
        # scale the data! implement MinMax scaling for each of the three dimensions to be in range -1,1
        # LORENZ MinMax Scaling
        # self.data[:,0] *= 0.05
        # self.data[:,1] *= 0.05
        # self.data[:,2] *= 0.02

        # DG model MinMax Scaling
        # self.data[:,0] = self.minMaxScaler(self.data[:,0]) 
        # self.data[:,1] = self.minMaxScaler(self.data[:,1])

        # Initialize fixed bias (for double gyre, we know that expected range x: (0,2), y: (0,1) )
        # self.data[:,0] += -1.0
        # self.data[:,1] += -0.5



        # Initilize weights
        # self.Win = (np.random.rand(self.resSize, self.inSize+1)-0.5) * self.input_scaling  # (resSize x (inSize+1) ) added bias column
        self.Win = np.multiply( self.getRandomSparseMatrix((self.resSize, self.inSize+1), density=0.25), \
                                np.random.uniform(-1,1,(self.resSize, self.inSize+1)) )
        self.Wout = np.random.rand(self.outSize,1)-0.5
        self.A = self.getRandomGraph(self.resSize)
        self.A = np.multiply(np.random.uniform(-0.5,0.5,(self.resSize, self.resSize)), self.A)
        self.P = np.random.rand(self.outSize, self.resSize)-0.5

        # initialize states
        # adding bias, input to state matrix r
        self.r = np.zeros((1+self.inSize+self.resSize, self.trainLen-self.initLen+1)) 

        # Initialize data vectors 
        self.u = self.data[self.initLen : self.trainLen]
        self.v_tgt = self.data[ self.initLen+1 : self.trainLen+1]

        # self.v = np.zeros((self.outSize, self.trainLen-self.initLen))
        # Add noise to data (u) vector
        # self.input_noise = np.random.normal(0,0.01,self.u.shape)
        # self.u += self.input_noise

        # Normalize reservoir matrix A by spectral radius
        rhoW = max(abs(linalg.eig(self.A)[0]))
        self.A *= self.spectral_radius/rhoW

        # Sample training states (r) data
        self.r_t = np.zeros((self.resSize,1)) 
        for t in range(self.trainLen-self.initLen):
            self.r_t = self.leaking_rate*self.r_t \
                            + (1.0-self.leaking_rate)*np.tanh( np.dot(self.Win, np.vstack((1, self.u[t].reshape(self.inSize,1))) ) \
                                                             + np.dot( self.A, self.r_t) ) \
                            + self.noise*np.random.rand(self.resSize,1) #same shape as r_t
            self.r[:,t+1] = np.vstack((1,self.u[t].reshape(self.inSize,1), self.r_t)).reshape(self.inSize+self.resSize+1)

        # initialize PREDICTION reservoir state variable (to be used for esn.predict() method )
        self.predict_r_t_ = np.zeros((self.resSize, 1))  # initial PREDICTION reservoir state r_t

        # clean up variables 
        del( self.data )
        del( self.r_t )
        del( self.u )

        return 

    def train(self):
        """Train on training states using linear ridge regression
        
        Find P s.t. minimizes
        ||v_d - Pr||^2 + a*||P||^2
        
        """
        logging.info("Training...")
        
        self.P = np.dot(np.dot(self.v_tgt.T,self.r[:,1:].T), linalg.inv(np.dot(self.r[:,1:],self.r[:,1:].T) + \
            self.ridgeReg*np.eye(1+self.inSize+self.resSize) ) )


        # import pdb;pdb.set_trace()

        return

    def test(self, testing_data):
        """Test on actual data using trained model"""
        logging.info("Testing...")
        # import pdb;pdb.set_trace()

        # Initialize test data vectors
        self.testLen = testing_data.shape[0]
        self.u_ = testing_data[:1]  # Just one data point to kick off
        self.v_tgt_ = testing_data[1 : self.testLen]

        
        # Initialize test states
        self.r_ = np.zeros((1+self.inSize+self.resSize, self.testLen))
        self.v_ = np.zeros((self.outSize, self.testLen))

        ##################
        self.r_t_ = self.r[1+self.inSize:,-1].reshape(self.resSize, 1)
        # self.r_t_ = np.zeros((self.resSize,1)) 
        ######################

        self.v_[:,0] = self.u_
        self.r_[:,0] = np.vstack((1, self.u_.T, self.r_t_ )).T # 1 x 1+inSize+dimSize


        # Sample test states (r) data
        # Generate estimate states and values
        for t in range(self.testLen-1):
            self.r_t_ = self.leaking_rate*self.r_t_ \
                            + (1.0-self.leaking_rate)*np.tanh( np.dot(self.Win, np.vstack((1, self.u_.T)) )
                                                                + np.dot(self.A, self.r_t_) ) \
                            + self.noise*np.random.rand(self.resSize,1) #same shape as r_t_
            
            self.r_[:,t+1] = np.vstack((1, self.u_.T, self.r_t_ )).T
            self.v_[:,t+1] = np.dot( self.P, np.vstack((1, self.u_.T, self.r_t_ ))).reshape(self.outSize)

            if self.partial_know:
                # Forcing one dimension to be actual
                self.v_[1,t+1] = self.v_tgt_.T[1,t+1]
            self.u_ = self.v_[:,t+1].T.reshape(1,self.outSize)

        # clean up variables 
        del( self.r_t_ )
        self.v_ = np.delete(self.v_,0,1)

        # import pdb;pdb.set_trace()
        return 

    def predict(self, input_us, res_state=None):
        """
        input_u : input_sizexself.inSize ( e.g. np.array( [[1,2], [1.1,2.2] ))
        res_state : (optional) res_sizex1  (used to consequtive predictions that require previous reservoir state)
        Return t+1 step from input_u based on trained model 
        """
        # input_us[:,0] += -1.0
        # input_us[:,1] += -0.5

        if res_state :
            self.predict_r_t_ = res_state
        else:
            self.predict_r_t_ = np.zeros((self.resSize, 1))  # random initial PREDICTION reservoir state r_t

        output_v = np.zeros((np.shape(input_us)[0], self.outSize))
        for i, input_u in enumerate(input_us): 
            # r_t_ = np.zeros((self.resSize,1))  # initial state r_t
            self.predict_r_t_ = self.leaking_rate*self.predict_r_t_ \
                                + (1.0-self.leaking_rate)*np.tanh( np.dot(self.Win, np.vstack((1, input_u.reshape(1,self.inSize).T)) )
                                                                    + np.dot(self.A, self.predict_r_t_) ) \
                                + self.noise*np.random.rand(self.resSize,1) #same shape as r_t_

            v_t_1 = np.dot( self.P, np.vstack((1, input_u.reshape(1,self.inSize).T, self.predict_r_t_ ))).reshape(self.outSize)
            output_v[i,:] = v_t_1

        # output_v[:,0] += 1.0
        # output_v[:,1] += 0.5
        return output_v

    def plot(self, length, name, show=False, only_test=False, dim=2):
        plt.figure()
        # coorMap = {0: 'x', 1: 'y', 2: 'z'}
        for coor in range(dim):
            plt.subplot(dim,1,coor+1)
            self.v = np.dot(self.P, self.r[:,1:]) # get estimates for training period 
            if only_test: # only plot testing phase
                plt.plot(self.v_tgt_[0:int(length)][:,coor], label='Target')
                plt.plot(self.v_.T[0:int(length)][:,coor], label='Estimate')
            else:
                plt.plot(np.concatenate( (self.v_tgt[-int(length):-1], self.v_tgt_[0:int(length)] ) )[:,coor], label='Target')
                plt.plot(np.concatenate( (self.v.T[-int(length):-1], self.v_.T[0:int(length)] ) )[:,coor], label='Estimate')
                plt.axvline(x=min(int(length),self.v.shape[1]), color='k', linestyle='--')
            # plt.ylim(-15,15)
            plt.ylabel(coor+1).set_rotation(0)
            if coor==0:
                plt.title(name)
                plt.legend()

        if show:
            plt.show()
        else:
            plt.savefig('{}.png'.format(name))
            plt.close('all')

    def plotStates(self,length,name,show=False):
        plt.figure()
        plt.plot(np.concatenate(( self.r.T[-int(length):-1], self.r_.T[0:int(length)] )).T[::20].T )   # plot sample of states
        plt.axvline(x=min(int(length),self.r.shape[1]), color='k', linestyle='--')
        plt.title('States')
        if show:
            plt.show()
        else:
            plt.savefig('{}_states.png'.format(name))
            plt.close('all')

    def plotWeights(self,name,show=False):
        plt.figure()
        plt.plot(self.P.T)
        plt.title('Weights (P)')
        plt.legend()
        if show:
            plt.show()
        else:
            plt.savefig('./results/{}_weights.png'.format(name))
            plt.close('all')

    def plotLorenzMap(self, dim=0):
        from scipy.signal import argrelextrema

        # get only relevant dimension (y?)
        predict = abs(self.v_[dim])
        actual = abs(self.v_tgt_.T[dim])

        # get local maxima
        predict_maxima = predict[argrelextrema(predict, np.greater)[0]]
        actual_maxima = actual[argrelextrema(actual, np.greater)[0]]

        # create arrays for successive maxima
        getSuccessive = lambda a: np.array(list(map(list, zip(a[:-1], a[1:]))))
        successive_predict = getSuccessive(predict_maxima)
        successive_actual = getSuccessive(actual_maxima)

        # plot 
        x_, y_ = successive_predict.T
        x, y = successive_actual.T
        plt.scatter(x,y)
        plt.scatter(x_,y_)

        # plt.xlim(-1,1)
        # plt.ylim(-0.5,0.5)
        plt.show()
    
if __name__ == '__main__':
    # main()

    """
    Notes:
    useful - A Practical Guide to ESNs <http://minds.jacobs-university.de/uploads/papers/PracticalESN.pdf>
    """

    str_datetime = datetime.now().__str__().replace('-','') \
                                           .replace(' ','_') \
                                           .replace(':','_') \
                                           .replace('.','') 

    train_input_data_fullpath = '../inputs/preprocess/QGds02di02dm02p3.TRAIN'
    train_flattened_input_data = util.flatten_time_series( util.load_data( train_input_data_fullpath ) ).transpose()
    esn = EchoStateNetwork(loaddata=train_flattened_input_data,initLen=0, resSize=100,partial_know=False,noise=0.01, 
                density=1e-2,spectral_radius=1.8,leaking_rate=0.2, input_scaling=0.3, ridgeReg=0.01,mute=False)

    
    esn.train()
    

    test_input_data_fullpath = '../inputs/preprocess/QGds02di02dm02p3.TEST'
    test_flattened_input_data = util.flatten_time_series( util.load_data( test_input_data_fullpath ) ).transpose()

    import pdb;pdb.set_trace()
    esn.test(testing_data=test_flattened_input_data[:1000])
    pdb.set_trace()
    esn.plot(length=1000, name='{}'.format(str_datetime), show=True)
    # esn.plot(length=10000, name='{}'.format(str_datetime), show=False)
    esn.plotLorenzMap()
    esn.plotStates(length=2000)
    esn.plotWeights()
    # print('Plotting Lorenz map for x...')
    # esn.plotLorenzMap(dim=0)
    # print('Plotting Lorenz map for y...')
    # esn.plotLorenzMap(dim=1)

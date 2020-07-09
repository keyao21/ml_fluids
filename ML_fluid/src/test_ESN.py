import unittest
import ESN
import util
import numpy as np 


class Test_ESN(unittest.TestCase):
    def test_ESN_test(self): 
        # generate fake data 
        fakedata = np.random.rand(10,4)
        resSize = 10 
        partial_know = False 
        noise = 0 
        density = 0.5
        spectral_radius = 1.0
        leaking_rate = 0.2
        input_scaling = 0.2
        ridgeReg = 0.1
        mute = False 
        testesn = ESN.EchoStateNetwork(loaddata=fakedata, resSize=resSize, partial_know=partial_know, noise=noise, density=density, 
                                        spectral_radius=spectral_radius, leaking_rate=leaking_rate, input_scaling=input_scaling, 
                                        ridgeReg=ridgeReg, mute=mute)
        testesn.train()

        faketestdata = np.random.rand(10,4)
        testesn.test(testing_data=faketestdata)
        testsfest = testesn.v_.reshape(2,2,-1)
        testsfactual = testesn.v_tgt.transpose().reshape(2,2,-1)
        self.assertEqual( testsfactual.shape, testsfest.shape )
        self.assertEqual( testsfactual.shape[-1], 10-1)
    # def test_flatten_time_series(self): 
    #     # check basic functionality
    #     test_array = np.random.rand(3, 4, 10)
    #     flattened_test_array = util.flatten_time_series( test_array )
    #     self.assertEqual( flattened_test_array.shape, (12,10)  )
    #     # check warning logging
    #     with self.assertLogs(level='INFO') as cm:
    #         test_array_reordered = np.random.rand(10, 3, 4)
    #         flattened_test_array_reordered = util.flatten_time_series( test_array_reordered )
    #         self.assertEqual( flattened_test_array.shape, (12,10)  )
    #     self.assertEqual(cm.output, [ 'WARNING:root:Last dimension is not the largest, check order of dimensions' ])

    # def test_split_training_testing(self): 
    #     data = np.random.rand(2,3,10)
    #     train_data, test_data = util.split_training_testing( data, training_length=4, axis=2) 
    #     self.assertEqual( train_data.shape, (2,3,4))
    #     self.assertEqual( test_data.shape, (2,3,6))
    #     self.assertRaises( Exception, util.split_training_testing, data=data, training_length=11, axis=2)
        
if __name__ == '__main__':
    unittest.main()
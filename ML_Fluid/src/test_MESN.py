import pdb
import numpy as np
import unittest
import MESN

class Test_MESN(unittest.TestCase):
    def test_MESN(self): 
        # generate fake data 
        fakedata = np.random.rand(10,4,2)

        resSize = 10 
        partial_know = False 
        noise = 0 
        density = 0.5
        spectral_radius = 1.0
        leaking_rate = 0.2
        input_scaling = 0.2
        ridgeReg = 0.1
        mute = False 
        testesn = MESN.MultiEchoStateNetwork(loaddata=fakedata, resSize=resSize, partial_know=partial_know, noise=noise, density=density, 
                                        spectral_radius=spectral_radius, leaking_rate=leaking_rate, input_scaling=input_scaling, 
                                        ridgeReg=ridgeReg, mute=mute)
        testesn.train()
        # import pdb;pdb.set_trace()
        faketestdata = np.random.rand(10,4)
        testesn.test(testing_data=faketestdata)
        testsfest = testesn.v_.reshape(2,2,-1)
        testsfactual = testesn.v_tgt_.transpose().reshape(2,2,-1)
        # import pdb;pdb.set_trace()
        self.assertEqual( testsfactual.shape, testsfest.shape )
        self.assertEqual( testsfactual.shape[-1], 10-1)

if __name__ == '__main__':
    unittest.main()
import unittest 

import numpy as np 
import numpy.testing as nptest
from lib.integrators import * 

def acc_zero(x, v, t):
    return 0

class TestEulerIntegrator(unittest.TestCase):
    
    def setUp(self):
        self.integrator = EulerIntegrator()

    def test_constant(self):
        x0 = 5
        v0 = 0
        x,v = self.integrator.integrate(x0, v0, acc_zero, 5, 0.01)
        nptest.assert_almost_equal(x[-1], 5)
        nptest.assert_almost_equal(v[-1], 0)
    
    def test_linear(self):
        x0 = 5
        v0 = 10
        x,v = self.integrator.integrate(x0, v0, acc_zero, 5, 0.01)
        nptest.assert_almost_equal(x[-1], 55)
        nptest.assert_almost_equal(v[-1], 10)


class TestRK4Integrator(unittest.TestCase):
    
    def setUp(self):
        self.integrator = RK4Integrator()

    def test_constant(self):
        x0 = 5
        v0 = 0
        x,v = self.integrator.integrate(x0, v0, acc_zero, 5, 0.01)
        nptest.assert_almost_equal(x[-1], 5)
        nptest.assert_almost_equal(v[-1], 0)
    
    def test_linear(self):
        x0 = 5
        v0 = 10
        x,v = self.integrator.integrate(x0, v0, acc_zero, 5, 0.01)
        nptest.assert_almost_equal(x[-1], 55)
        nptest.assert_almost_equal(v[-1], 10)


if __name__ == '__main__':
    unittest.main()

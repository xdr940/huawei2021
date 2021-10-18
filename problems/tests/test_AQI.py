
import unittest
import numpy as np
import os
from utils.evaluation import AQI_np2,AQI

class TestDescription(unittest.TestCase):

    def testAQI(self):
        self.assertEqual(AQI('SO2', 12),12)
        self.assertEqual(AQI('NO2', 66),83)
        self.assertEqual(AQI('CO',0.8),20)
        self.assertEqual(AQI('O3',210),146)
        self.assertEqual(AQI('PM10',83),67)
        self.assertEqual(AQI('PM2.5',39),55)

        print(AQI_np2('O3',np.array([185,210])))
        self.assertEqual(
            list(AQI_np2('O3',np.array([185,210]))),
            list(np.array([123,146]))
        )
import unittest
import sys
import os

pwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(pwd))
from uak import PEFeatureExtractor

class call_raw_features(unittest.TestCase): 
    # def test_raw_features(self):
    #     extractor = PEFeatureExtractor()
    #     path = '/home/choi/Downloads/TrainSet/00011da62b42afdb113732252d44b054.vir'
    #     extractor.raw_features(path)

    def test_feature_vector(self):
        extractor = PEFeatureExtractor()
        path = '/home/choi/Downloads/TrainSet/00011da62b42afdb113732252d44b054.vir'
        vector = extractor.feature_vector(path)
        print(vector.shape)

if __name__=='__main__':    
    unittest.main()
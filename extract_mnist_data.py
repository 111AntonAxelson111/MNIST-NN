
import numpy as np
from typing import Tuple, List
import csv


def extract_training_data_from_csv()->List[ Tuple[int,np.ndarray] ]:
    print("loading training data")
    train_data: List[ Tuple[int,np.ndarray] ]
    train_data = []
    
    with open('mnist_train.csv') as f:
        
        content = csv.reader(f)
        content = list(content)
        content = content[1:] #remove first line
        
        # make each image to an numpy array
        # first index represent the actual number of image, we call this label
        for i,img in enumerate(content):
            img = [int(pxl) for pxl in img] #this not really only image
            label = img[0]
            img = np.array(img[1:],dtype=np.float32)  
            
            #normalize
            img = img /255.0
            train_data.append((label,img))
    
    
    
    
    print("done loading train data")
    print()
    
    return train_data

def extract_test_data_from_csv()->List[ Tuple[int,np.ndarray] ]:
    print("loading test data")
    print()
    test_data: List[ Tuple[int,np.ndarray] ]
    test_data = []
    
    with open('mnist_test.csv') as f:
        
        content = csv.reader(f)
        content = list(content)
        content = content[1:] #remove first line
        
        # make each image to an numpy array
        # first index represent the actual number of image, we call this label
        for i,img in enumerate(content):
            img= [int(pxl) for pxl in img]  #this is label+image
            label = img[0]
            img = np.array(img[1:],dtype=np.float32)  
            
            #normalize
            img = img /255.0
            
            test_data.append((label,img))
    
    print("done loading test data","time:")
    print("-------------------------------------\n\n")
    
    return test_data



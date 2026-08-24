import numpy as np


class MNISTModel:
    
    
    def __init__(self, W1, b1, W2, b2):
        self.W1 = W1
        self.b1 = b1
        self.W2 = W2
        self.b2 = b2
    
    
    def __call__(self, X):
        
        hidden_raw = self.W1 @ X + self.b1 #weighted sum
        
        hidden_activated = np.maximum(0,hidden_raw) # relu
        
        output_raw = self.W2 @ hidden_activated + self.b2 #weighted sum
        
        exps = output_raw - np.max(output_raw) # softmax
        exps = np.exp(exps)   
        exps = exps / np.sum(exps)
        
        pred_y = int(np.argmax(exps))  # select biggest number from probability distrobution 
        
        return pred_y
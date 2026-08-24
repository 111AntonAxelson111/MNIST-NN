

"""   

* Structure of neural network

Input layer: 
Is a flattened 28×28 MNIST image.
its a 1d numpy array with shape (784,)
each element has a float32 value inbetween 0 to 1

Hidden lagret:
128 neurons
Relu 

Output layer:
10 neurons
softmax: o_i = (e**z_i)/(∑_j(e**z_j))
the output is a 1d numpy array with shape (10,)
each element has a float32 value inbetween 0 to 1
it represent a probability distrobution and the index whit highest value is the prediction
each index represents a value from 0 to 9  where first element is 0 and last is 9



* how N_EPOCHS effect accuracy and how much time training takes
this n wrong predictions for model per epoch with   

n wrong:  74 54 35 34 35 25 24 24 25 23 26 22 23 21 21
n epoch:  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15
time min: 1  3  5  8  10 12 15 17 19 22 24 26 29 31 33     

so n_epoch==3 for okay
so n_epoch==12 for good



* Other usefull info

function save_params_text  is not so usefull, you acn remove it

model.py is used to create an function that can be serialized. 
Also for it to work when its deserialized 





"""







import numpy as np
from typing import Tuple,List
import time
from icecream import ic
from easy_csv_logging import log
from numba import njit
import random
from datetime import datetime
import pickle
import os

from extract_mnist_data import extract_training_data_from_csv,extract_test_data_from_csv
from model import MNISTModel


print()
print()
print()
start_time = time.time()
print("starting_now","time:",time.time()-start_time)
print()


INPUT_SIZE = 784
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
LEARNING_RATE = 0.001 
N_EPOCHS = 3
SUBSET_N_ITER = 10
SMALL_TEST_BATCH_SIZE = 1000
DEBUG= False
OUT_DIR = "out"



# weights and  biases
W1 = np.random.randn(HIDDEN_SIZE, INPUT_SIZE).astype(np.float32) * 0.01
b1 = np.zeros(HIDDEN_SIZE).astype(np.float32)
W2 = np.random.randn(OUTPUT_SIZE, HIDDEN_SIZE).astype(np.float32) * 0.01
b2 = np.zeros(OUTPUT_SIZE).astype(np.float32)



def get_timestamp()->str:
    return datetime.now().strftime("%Y_%m_%d__%H_%M_%S")

@njit  
def softmax(arr: np.ndarray) -> np.ndarray:
    exps = arr - np.max(arr)
    exps = np.exp(exps)   #[ e^arr_i for arr_i in arr]
    exps = exps / np.sum(exps)
    return exps

@njit
def forward(
        flat_img: np.ndarray,
        W1: np.ndarray,
        b1: np.ndarray,
        W2: np.ndarray,
        b2: np.ndarray   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    hidden_raw = W1 @ flat_img + b1
    
    hidden_activated = np.maximum(0, hidden_raw) #== relu(hidden_raw) # Hidden activated layer
    
    output_raw = W2 @ hidden_activated +b2 # Output raw layer
    
    output = softmax(output_raw) # Softmax output
    
    return hidden_raw, hidden_activated, output_raw, output

@njit  # Backpropagation 
def train_step(label,img, W1, b1, W2, b2):
    target = np.zeros(10, dtype=np.float32)
    target[label] = 1.0

    hidden_raw, hidden, output_raw, output = forward(img, W1, b1, W2, b2)

    error_output = output - target

    dW2 = np.outer(error_output, hidden)
    db2 = error_output

    error_hidden = (W2.T @ error_output) * (hidden_raw > 0)

    dW1 = np.outer(error_hidden, img)
    db1 = error_hidden

    W2 -= LEARNING_RATE * dW2   # om dW2==0 -> ingen adjustment 
    b2 -= LEARNING_RATE * db2
    W1 -= LEARNING_RATE * dW1
    b1 -= LEARNING_RATE * db1

    return W1, b1, W2, b2

@njit
def predict(img, W1,b1,W2,b2):
    _, _, _, output = forward(img, W1,b1,W2,b2)
    return np.argmax(output)

def test_accuracy(
                W1:  np.ndarray,
                b1:  np.ndarray,
                W2:  np.ndarray,
                b2:  np.ndarray,
                test_batch: List[Tuple[int,np.ndarray]]
                )->str:
    
    result = ""
    
    n_correct = 0 
    n_NOT_correct = 0 
    
    for label,img in test_batch:
        
        y_pred = predict(img,W1,b1,W2,b2)
        
        if label==y_pred:
            n_correct+=1
        else:
            n_NOT_correct+=1
    
    result = f"\n====test====\nCORRECT:{n_correct}\nNOT CORRECT:{n_NOT_correct}\n====test===="
    
    
    return result

def save_params_text(W1,b1,W2,b2):
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    
    mmm="======="
    time_stamp=get_timestamp()
    filename="params_"+time_stamp+".txt"
    path = os.path.join(OUT_DIR,filename)
    s="\n\n\n\n\n\n===================================="
    s+=f"{mmm} TIMESTAMP:{time_stamp} {mmm}\n"
    
    for name, X in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
        s += f"\n\n\n======= {name} =======\n"

        if X.ndim == 1:
            for val in X:
                s += f"{float(val):10.5f} "
            s += "\n\n"
        else:
            for row in X:
                s += " ".join(f"{float(v):10.5f}" for v in row) + "\n"
            s += "\n"

    
    with open(path,"w",encoding="utf-8") as f:
        f.write(s)

def save_ready_model(model_to_be_serialized, accuracy):
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    timestamp = get_timestamp()
    
    model_data = {
        "model": model_to_be_serialized,
        "timestamp": timestamp,
        "accuracy": accuracy
    }
    
    filename = (
        f"model_{timestamp}.pkl"
    )
    
    filepath = os.path.join(OUT_DIR,filename)
    
    with open(filepath, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"Ready model saved:")
    print(filepath)

    return filepath


train_data: List[ Tuple[int,np.ndarray] ]
train_data = extract_training_data_from_csv()

test_data: List[ Tuple[int,np.ndarray] ]
test_data = extract_test_data_from_csv()


# this is not really necissary, this saves model that is not trained
#
save_params_text(W1,b1,W2,b2)  # write params in  readable way to a textfile

# We save serialise current model as f(X)->Y with its accuracy
model_to_be_serialised = MNISTModel(W1, b1, W2, b2)
model_accuracy = test_accuracy(W1,b1,W2,b2,test_data)
save_ready_model(model_to_be_serialised,model_accuracy)
#


#we use this to quckly display how well it is performing
small_test_batch = test_data[0:SMALL_TEST_BATCH_SIZE]


# subset thing we do
subset= []
subset_conter=0

for epoch in range(N_EPOCHS):
    
    print("\n\n=============================================")
    print("start epoch ",epoch, "   ","time:",time.time()-start_time)
    
    random.shuffle(train_data) # each epoch we change order of the dataset 
    
    n_images_counter=0
    for label,img in train_data:
        
        n_images_counter+=1
        W1,b1,W2,b2 = train_step(label,img,W1,b1,W2,b2)
        """ 
        This following is not really necissary.
        but if its removed number of epochs needs to be changed
        """
        
        subset_conter+=1
        if subset_conter>20: 
            subset.append((label,img)) 
        if subset_conter>30:
            
            for _ in range(SUBSET_N_ITER): 
                for label,img in subset:  #here we loop over batch and train on the sequence of images SUBSET_N_ITER number of times
                    W1,b1,W2,b2 = train_step(label,img,W1,b1,W2,b2)
            
            subset_conter=0  # reset
            subset=[]
        
    
    small_test_result = test_accuracy(W1,b1,W2,b2,small_test_batch)
    print(small_test_result)
    print("number of images",n_images_counter)





# save model now
save_params_text(W1,b1,W2,b2)  # this not really necissary


# We serialize model and write it to the path out with its accuracy
model_accuracy = test_accuracy(W1,b1,W2,b2,test_data)
model_to_be_serialised = MNISTModel(W1, b1, W2, b2)
save_ready_model(model_to_be_serialised,model_accuracy)





print("testing current model")
print("actual\t"+"predicted")
iii=0
for y_actual,flat_img in train_data:
    iii+=1
    if iii>30:
        iii=0
        input("...")
        print()
    
    _,_,_,out = forward(flat_img, W1, b1, W2, b2)
    print(str(y_actual)+"\t"+str(np.argmax(out)))
    print("-----------------------\n")
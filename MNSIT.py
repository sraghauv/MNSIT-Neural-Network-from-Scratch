
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def data_initalization(m):
    data_set = pd.read_csv('/Users/raghauvsaravanan/Desktop/Datasets/digit-recognizer/train.csv')
    data_set = np.array(data_set)
    np.random.shuffle(data_set)
    data_set = data_set.T
    labels = data_set[0, 0:m]
    
    one_hot = one_hot_encode(labels)
    
    x_train = data_set[1:, 0:m]
   
    x_train = x_train / 255
    
    return x_train, one_hot, labels

def data_test_initalization(m):
    data_set = pd.read_csv('/Users/raghauvsaravanan/Desktop/Datasets/digit-recognizer/train.csv')
    data_set = np.array(data_set)
    np.random.shuffle(data_set)
    data_set = data_set.T
    labels = data_set[0, 1000:1000 + m]
    
    one_hot = one_hot_encode(labels)
    x_train = data_set[1:, 1000:1000+m]
    
    x_train = x_train / 255
    
    return x_train, one_hot, labels
    
    

def one_hot_encode(labels):
    y_hot = np.zeros((labels.size, 10 ))
    y_hot[np.arange(labels.size),labels] = 1
    y_hot = y_hot.T
    return y_hot

def randomize_paramters(node_size,m):
    w1 = .01 * np.random.rand(node_size, 784)
    w2 = .01 * np.random.rand(10,node_size)
    b1 = np.zeros((node_size,m))
    b2 = np.zeros((10,m))
    return w1, w2, b1, b2

def reLU(x):
    return np.maximum(0,x)

def delta_reLU(x):
    return (x > 0) * 1

def soft_max(Z):
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)


def forward(w1,w2,b1,b2,x):
    z1 = np.dot(w1, x) + b1
    a1 = reLU(z1)
    z2 = np.dot(w2,a1) + b2
    a2 = soft_max(z2)
    
    return z1, a1, a2

def back_prop(a1, a2, z1, w2,one_hot,m,x):
    
    delta_loss = a2 - one_hot
    
    dW2 = (1 / m)  * np.dot(delta_loss, a1.T)
    dB2 = (1 / m)  * delta_loss
    dZ1 = np.dot(w2.T, delta_loss) * delta_reLU(z1)
    dW1 = (1 / m) * np.dot(dZ1,x.T)
    dB1 = (1 / m) * dZ1
    
    return dW2, dB2, dW1, dB1

def implement_loss(w1, w2, b1, b2, dW2, dB2, dW1, dB1, learning_rate):
    w1 -= dW1 * learning_rate
    w2 -= dW2 * learning_rate
    b1 -= dB1 * learning_rate
    b2 -= dB2 * learning_rate
    return w1, w2, b1, b2

def terminal():
    m = int(input("Dataset Size: "))
    hidden_layer = int(input("Hidden Layer Size: "))
    iterations = int(input("Iterations: "))
    learning_rate = float(input("Learning Rate: "))
    return m, hidden_layer, iterations, learning_rate




    
def get_predictions(A2):
    return np.argmax(A2,0)

def get_accuracy(predictions, Y):
    
    return np.sum(predictions == Y) / Y.size        
   
    
    
        

def gradient_descent(x, one_hot, m, hidden_layer_size, iterations, learning_rate, labels):
    w1, w2, b1, b2 = randomize_paramters(hidden_layer_size,m)
    
    count = 0
    for i in  range(iterations):
        z1, a1, a2 = forward(w1, w2, b1, b2, x)
        dW2, dB2, dW1, dB1 = back_prop(a1, a2, z1, w2, one_hot, m, x)
        w1, w2, b1, b2 = implement_loss(w1, w2, b1, b2, dW2, dB2, dW1, dB1, learning_rate)
        if i % 100 ==0:
            print("Iteration ",i)
            
            
    return w1, w2, b1, b2
       
        
   
def test(w1, w2, b1, b2,x,labels,m):
        z1, a1, a2 = forward(w1, w2, b1, b2, x)
        
        count = 0
        for i in range (m):
            if labels[i] == get_predictions(a2)[i]:
                count += 1
                
        accuracy = (count / m) 
        print("Accuracy ", accuracy)
            
        

        
def image_view(w1, w2, b1, b2, x_test, labels_test,m):
    response = input("Would you like to view images? (y/n)")
    
    if(response=="y"):
        while True:
            _, _, a2 = forward(w1, w2, b1, b2, x_test)
            
            index = int(input("Enter a number (0 - " + str(m-1) + "):"))
            
            image = x_test[:, index].T
            print("Prediction: ", (get_predictions(a2)[index]))
            print("Label: ", labels_test[index])
            
            image = image.reshape((28,28)) * 255
           
            plt.gray()
            
            plt.imshow(image, interpolation = 'nearest')
            plt.show()
            
        
        
    
        
    
    
def main():
    m, hidden_layer_size,iterations, learning_rate = terminal()
    x_train, one_hot, labels = data_initalization(m)
    x_test, one_hot_test, labels_test = data_test_initalization(m)
    w1, w2, b1, b2 = gradient_descent(x_train, one_hot, m, hidden_layer_size, iterations, learning_rate, labels)
    test(w1, w2, b1, b2, x_test, labels_test,m)
    image_view(w1, w2, b1, b2, x_test, labels_test,m)
    print("Program Complete")
    
    
    
    
    
    
main()








'''
def terminal():
    m = int(input("How large would you like your data set: "))
    num_hidden_layers = int(input("How many hidden layers would you like? "))
    hidden_layer_sizes = []
    for i in range (num_hidden_layers):

        print("Hidden Layer Layer ", i+1)
        hidden_layer_sizes.append(int(input("What size would you like for this layer? ")))
        
    return m, hidden_layer_sizes
    
   def arg_max(x):
    
    y = np.argmax(x)
    return y


def accuracy(count, m):
    return (count / m) * 100
    
'''   
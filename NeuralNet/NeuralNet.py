import numpy as np

import matplotlib.pyplot as plt

np.random.seed(42)




# generate data
var = 0.2
n = 800
class_0_a = var * np.random.randn(n//4,2)
class_0_b =var * np.random.randn(n//4,2) + (2,2)

class_1_a = var* np.random.randn(n//4,2) + (0,2)
class_1_b = var * np.random.randn(n//4,2) +  (2,0)

X = np.concatenate([class_0_a, class_0_b,class_1_a,class_1_b], axis =0)
Y = np.concatenate([np.zeros((n//2,1)), np.ones((n//2,1))])
X.shape, Y.shape



# shuffle the data
rand_perm = np.random.permutation(n)

X = X[rand_perm, :]
Y = Y[rand_perm, :]



# train test split
ratio = 0.8
X_train = X [:int (n*ratio)]
Y_train = Y [:int (n*ratio)]

X_test = X [:int (n*ratio)]
Y_test = Y [:int (n*ratio)]



plt.scatter(X_train[:,0], X_train[:,1], c=Y_train)
plt.show()






class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        
        self.weights = np.random.normal(0,2/(n_inputs+n_neurons),(n_inputs,n_neurons))
        
        self.biases = np.ones((1,n_neurons))
        

    
    
    def forward(self,X):
        
        self.output = np.dot(X , self.weights)+ self.biases
        
class Activation_ReLu:
    def forward(self,x):
        self.output = np.maximum(0,x)

class Activation_Sigmoid:
    def forward(self,x):
        self.output = 1/(1+np.exp(-x))

class Activation_softmax:
    
    
    
    
    def forward(self,x):
        exp_values = np.exp(x-np.max(x, axis = 1, keepdims = True))
        probabilites  =  exp_values / np.sum(exp_values,axis = 1, keepdims = True)
        self.output = probabilites

def LCE(y_pred,y_true):
    
    
    # loss = -np.sum(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
    loss = -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))
    
    return loss

Layer1 = Layer_Dense(n_inputs=2, n_neurons=7)
Activation1 = Activation_Sigmoid()




Layer2 = Layer_Dense(n_inputs=7, n_neurons=1)
Activation2 = Activation_Sigmoid()



 # The back propagation
def backward_pass(X,Y, A2,Z2, A1,Z1 ,W1, W2, b1, b2):



  dl_dA2 = -((Y-A2)/(A2*(1-A2)))/Y.shape[0]


  dA2_d_Z2 = (A2*(1-A2))


  # dZ2_dA1 = W2.T

  dA1_dZ1 = A1*(1-A1)

  dW1 = X_train.T@((dl_dA2*dA2_d_Z2)@W2.T*dA1_dZ1)


  dW2 = ( dl_dA2 * dA2_d_Z2 ).T @ A1

  # one = np.ones((Z1.shape))
  db1 = ((dl_dA2*dA2_d_Z2).T@dA1_dZ1)*W2.T

  db2 = ( dl_dA2 .T@ dA2_d_Z2 )




  return dW1 , dW2 , db1 , db2 




# dW1 , dW2 , db1 , db2 = backward_pass(X_train,Y_train, A2, Z2, A1, Z1, W1, W2, b1, b2)



def update(W1, W2, b1, b2,dW1, dW2, db1, db2, alpha ):

  # Your code here
  W1 = W1-alpha*dW1
  W2 = W2 -alpha*dW2
  b1 = b1 -alpha*db1
  b2 = b2 - alpha*db2

  return W1, W2, b1, b2







alpha = 5*10e-5
W1, W2, b1, b2 = Layer1.weights, Layer2.weights , Layer1.biases , Layer2.biases
n_epochs = 10000

train_loss = []
for i in range(n_epochs):
    
    #predict
    Layer1.forward(X_train)
    Z1 = Layer1.output
    Activation1.forward(Z1)
    A1 = Activation1.output


    Layer2.forward(A1)
    Z2 = Layer2.output
    Activation2.forward(Z2)
    A2 = Activation2.output
    
    
    
    y_pred = A2
    
    
    Loss = LCE(y_pred,Y_train)
    train_loss.append(Loss)
    
    if i%10 == 0:
        print(Loss)
        
        
    dW1 , dW2 , db1 , db2 = backward_pass(X_train,Y_train, A2, A1,Z2,Z1, W1, W2, b1, b2)
    
    W1, W2, b1, b2  = update(W1, W2, b1, b2,dW1, dW2, db1, db2, alpha )
    Layer1.weights, Layer2.weights , Layer1.biases , Layer2.biases  = W1, W2, b1, b2
plt.plot(train_loss)
plt.show()




def predict(X_train):
    Layer1.forward(X_train)
    Z1 = Layer1.output
    Activation1.forward(Z1)
    A1 = Activation1.output


    Layer2.forward(A1)
    Z2 = Layer2.output
    Activation2.forward(Z2)
    A2 = Activation2.output
    
    
    
    return [1 if s > 0.5   else 0 for s in A2 ]


y_pred = predict(X_test)

def accuracy(y_pred, y_true):
    
    return np.mean(y_pred==y_true)


print("accuracy=",accuracy(y_pred, Y_test))

import numpy as np

import matplotlib.pyplot as plt





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




X = X.T
Y = Y.T
X.shape, Y.shape



# train test split
ratio = 0.8
X_train = X [:, :int (n*ratio)]
Y_train = Y [:, :int (n*ratio)]

X_test = X [:, int (n*ratio):]
Y_test = Y [:, int (n*ratio):]


plt.scatter(X_train[0,:], X_train[1,:], c=Y_train[0,:])
#plt.show()






def forward_pass(X, W1,W2, b1, b2):
  Z1 = W1.dot(X) + b1
  A1 = 1/(1+np.exp(-Z1))
  Z2 = W2.dot(A1) + b2
  A2 = 1/(1+np.exp(-Z2))
  return A2, Z2, A1, Z1







def backward_pass(X,Y, A2, Z2, A1, Z1, W1, W2, b1, b2):
    
  # Your code here
  dl_dz2 = (A2-Y)/Y.shape[1]


  dl_dz1 = (W2.T)@dl_dz2*(A1*(1-A1))




  dW1 =  dl_dz1@X.T


  dW2 =  dl_dz2@A1.T


  db1 = dl_dz1@np.ones(Y.shape).T

  db2 = (dl_dz2)@np.ones(Y.shape).T

  return dW1, dW2, db1, db2



def loss(y_pred, Y):
    
  # Your code here

  return  -np.mean(Y*np.log(y_pred)+(1-Y)*np.log(1-y_pred))


def init_params():
  
  h0, h1, h2 = 2, 10, 1

  W1 = 0.5*np.random.randn(h1,h0)

  W2 = 0.5*np.random.randn(h2,h1)

  b1 = np.zeros((h1,1))

  b2 = np.zeros((1,1))

  return W1, W2, b1, b2



def plot_decision_boundary(W1, W2, b1, b2):
    x = np.linspace(-0.5, 2.5,100 )
    y = np.linspace(-0.5, 2.5,100 )
    xv , yv = np.meshgrid(x,y)
    xv.shape , yv.shape
    X_ = np.stack([xv,yv],axis = 0)
    X_ = X_.reshape(2,-1)
    A2, Z2, A1, Z1 = forward_pass(X_, W1, W2, b1, b2)
    plt.figure()
    plt.scatter(X_[0,:], X_[1,:], c= A2)
    plt.show()



def update(W1, W2, b1, b2,dW1, dW2, db1, db2, alpha ):
    
  # Your code here
  W1 = W1-alpha*dW1
  W2 = W2 -alpha*dW2
  b1 = b1 -alpha*db1
  b2 = b2 - alpha*db2

  return W1, W2, b1, b2










class Model: 
   def __init__(self):
         self.train_loss = []
         self.test_loss = []
         self.W1 = None
         self.W2 = None
         self.b1 = None
         self.b2 = None
    
 

   def fit(self,X_train, Y_train,alpha = 0.1,n_epochs = 10000):
        

        self.W1, self.W2, self.b1, self.b2 = init_params()
        for i in range(n_epochs):
            ## forward pass
            A2, Z2, A1, Z1 =  forward_pass(X_train, self.W1,self.W2, self.b1, self.b2)
            ## backward pass
            dW1, dW2, db1, db2 = backward_pass(X_train,Y_train, A2, Z2, A1, Z1, self.W1, self.W2, self.b1, self.b2)
            ## update parameters
            self.W1, self.W2, self.b1, self.b2 = update(self.W1, self.W2, self.b1, self.b2,dW1, dW2, db1, db2, alpha )

            ## save the train loss
            self.train_loss.append(loss(A2, Y_train))
            ## compute test loss
            A2, Z2, A1, Z1 = forward_pass(X_test, self.W1, self.W2, self.b1, self.b2)
            self.test_loss.append(loss(A2, Y_test))

            ## plot boundary
            if i %1000 == 0:
                plot_decision_boundary(self.W1, self.W2, self.b1, self.b2)






   def predict(self,X):
        A2 = forward_pass(X, self.W1,self.W2, self.b1, self.b2)[0]
        return [1 if s > 0.5 else 0 for s in A2[0]]
        
   def accuracy(self,y_pred, y):
        return np.mean(y_pred ==y)

    ## plot train et test losses
   

classifier = Model()


classifier.fit(X_train,Y_train)

plt.plot(classifier.train_loss)
plt.plot(classifier.test_loss)
plt.show()






y_pred = classifier.predict(X_train)
train_accuracy = classifier.accuracy(y_pred, Y_train)
print ("train accuracy :", train_accuracy)

y_pred = classifier.predict(X_test)
test_accuracy = classifier.accuracy(y_pred, Y_test)
print ("test accuracy :", test_accuracy)
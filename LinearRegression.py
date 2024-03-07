import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)



# generate data
def generate_data(n= 1000):
  np.random.seed(0)
  x = np.linspace(-5.0, 5.0, n).reshape(-1,1)
  y= (29 * x + 30 * np.random.rand(n,1)).squeeze()
  x = np.hstack((np.ones_like(x), x))
  return x,y


# generate data
x,y= generate_data()
# check the shape
y = y.reshape(-1,1)
print ((x.shape,y.shape))



def split_data(x,y,train_perc=0.8):
  N=x.shape[0]
  train_size=round(train_perc*N)
  x_train,y_train=x[:train_size,:],y[:train_size]
  x_test,y_test=x[train_size:,:],y[train_size:]
  return x_train,y_train,x_test,y_test



x_train,y_train,x_test,y_test=split_data(x,y)
print(f"x_train:{x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}")



plt.scatter(x_train[:,1],y_train, marker="+")
plt.xlabel("feature")
plt.ylabel('target')
plt.show()

class LogisticRegression:
  ''' Class of linear Regression model'''
  def __init__(self):

    self.theta = None

    self.losses = None



  def linear(self, X, theta): # linear function
    ''' Linear function for the the forward pass
    args : X ----> A matrix of size NxD
         : theta ------> A vector of dimention  Dx1

    return : A vector of dimension Nx1 
       '''
    return X@theta


  def MSE(self,y_pred, y_true):


    return np.mean((y_pred-y_true)**2)


  def one_gradient_mse(self,X,y,theta):


    return X.T@(self.linear(X,theta)-y)
  def Initialization(self,D):

    return np.zeros(D).reshape(-1,1)

  def update(self, theta, lr, grad):

    return theta- lr*grad
  
  def shuffle_data(self,X,y):


    ''' return a shuffle data by indx'''

    indx = np.random.permutation(X.shape[0])

    return  X[indx] , y[indx]
  
  
  def fit(self, X,y, lr=0.005,n_epoch=100):


    ''' train the model to find the best parameters or the weigth'''

    n, d = X.shape
    self.theta = self.Initialization(d)


    epoch = 0

    avg_loss = float("inf")

    loss_tolerance = 0.01  
    

    self.losses = []


    while epoch < n_epoch and avg_loss> loss_tolerance:
      
      runing_loss = 0.0

      X , y = self.shuffle_data(X,y) # shuflle the data


      for idx in range(n):

            xi =   X[idx].reshape(-1,d)
            yi =   y[idx].reshape(-1,1)


            yi_pred = self.linear(xi,self.theta)


            loss = self.MSE(yi,yi_pred)

            runing_loss = runing_loss+loss

            grad = self.one_gradient_mse(xi,yi,self.theta)


            self.theta = self.update(self.theta,lr,grad)
      avg_loss = runing_loss/n
      if epoch%5 ==0:
        print(f"at epoch = {epoch}, loss = {avg_loss}")
 
      self.losses.append(avg_loss)

      if np.linalg.norm(grad)<0.1:
        break
      epoch +=1



  def predict(self,X):

    return self.linear(X, self.theta)




model = LogisticRegression()


model.fit(x_train,y_train,n_epoch=10000)

loss =  model.losses

def plot_loss(loss):
  plt.plot(loss)
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.show()


plot_loss(loss)




    
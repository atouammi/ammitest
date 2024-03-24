import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)





class Linear_regression:
  '''Class of linear regression with simple gradient descent'''

  def __init__(self):
    self.theta = None

    self.losses = []



  def linear(self, X, theta): # linear function

    return X@theta


  def MSE(self,y_pred, y_true):


    return np.mean((y_pred-y_true)**2)


  def gradient_mse(self,X,y,theta):


    return (1/X.shape[0])*X.T@(self.linear(X,theta)-y)
  def Initialization(self,D):

    return np.zeros(D).reshape(-1,1)

  def update(self, theta, lr, grad):

    return theta- lr*grad

  def fit(self, X,y, lr=0.01,n_epoch = 1000):

    n,d = X.shape
    self.theta = self.Initialization(d)


    for epoch in range(n_epoch):


      y_pred = self.linear(X,self.theta)


      loss = self.MSE(y,y_pred)

      self.losses.append(loss)
    

      if epoch%100 ==0:
        print(f" at epoch = {epoch}, loss = {loss}")

      grad = self.gradient_mse(X,y,self.theta)
      if np.linalg.norm(grad) < 0.001:
        break

      self.theta = self.update(self.theta,lr,grad)


  def predict(self, X):

    return self.linear(X,self.theta)



if __name__ == "__main__":
      
      Linear_regression()



  

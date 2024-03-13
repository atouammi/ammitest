import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)




class LinearRegressionMm:
  

  def __init__(self):

    self.theta = None

    self.losses = None



  def linear(self, X, theta): # linear function

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
  def momentum(self,moment,beta,grad):


    return moment*beta + (1-beta)*grad
  
  def fit(self, X,y,lr=0.005,beta=0.10,n_epoch=100):


    ''' train the model to find the best parameters or the weigth'''

    n, d = X.shape
    self.theta = self.Initialization(d)

    
    epoch = 0
    
    avg_loss = float("inf")

    loss_tolerance = 0.01  
    

    self.losses = []
    


    while epoch < n_epoch and avg_loss> loss_tolerance:
      moment = 0
      
      runing_loss = 0.0
      X , y = self.shuffle_data(X,y) # shuflle the data
      

      for idx in range(n):

            xi =   X[idx].reshape(-1,d)
            yi =   y[idx].reshape(-1,1)


            yi_pred = self.linear(xi,self.theta)




            loss = self.MSE(yi,yi_pred)

            runing_loss = runing_loss+loss

            grad = self.one_gradient_mse(xi,yi,self.theta)

            moment = self.momentum(moment,beta,grad)

            self.theta = self.update(self.theta,lr,moment)
     
     
     
      avg_loss = runing_loss/n
      if epoch%5 ==0:
        print(f"at epoch = {epoch}, loss = {avg_loss}")
 
      self.losses.append(avg_loss)

      if np.linalg.norm(grad)<0.1:
        break
      epoch +=1



  def predict(self,X):

    return self.linear(X, self.theta)
  

if __name__ == "__main__":
  LinearRegressionMm()
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)




class LinearRegressionMb:
  

  def __init__(self):

    self.theta = None

    self.losses = None



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
  
  def shuffle_data(self,X,y):


    ''' return a shuffle data by indx'''

    indx = np.random.permutation(X.shape[0])

    return  X[indx] , y[indx]
  
  
  def fit(self, X,y, batch_size=10,lr=0.005,n_epoch=100):


    ''' train the model to find the best parameters or the weigth'''

    n, d = X.shape
    self.theta = self.Initialization(d)

    num_batch = n//batch_size


    X , y = self.shuffle_data(X,y) # shuflle the data
    

    self.losses = []


    for epoch in range(n_epoch):
      
      
      runing_loss = 0.0

    


      for idx in range(0,n,batch_size):

            x_batch =   X[idx:idx+batch_size].reshape(-1,d)
            y_batch =   y[idx:idx+batch_size].reshape(-1,1)


            yi_pred = self.linear(x_batch,self.theta)


            loss = self.MSE(y_batch,yi_pred)

            runing_loss = runing_loss+loss

            grad = self.gradient_mse(x_batch,y_batch,self.theta)


            self.theta = self.update(self.theta,lr,grad)
      avg_loss = runing_loss/num_batch
      if epoch%5 ==0:
        print(f"at epoch = {epoch}, loss = {avg_loss}")
 
      self.losses.append(avg_loss)

      if np.linalg.norm(grad)<0.15:
        break
      epoch +=1



  def predict(self,X):

    return self.linear(X, self.theta)



if __name__ == "__main__":
      LinearRegressionMb()
      
    












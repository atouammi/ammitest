import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)






x_train1,y_train1,x_test1,y_test1 = split_data_1(X_class,y_class)
y_train1 = y_train1.reshape(-1,1)


#plt.scatter(x_train1[:,0],x_train1[:,1],c = y_train1)



y_test1 = y_test1.reshape(-1,1)
print(f"x_train1:{x_train1.shape}, y_train1: {y_train1.shape}, x_test1: {x_test1.shape}, y_test1: {y_test1.shape}")



class Logistic_Regression:
  ''' Class of Logistic Regression model for binary case '''

  def __init__(self):
    self.theta = None
    self.losses = []



  def linear(self, X, theta): # linear function
      return X@theta


  def cross_entropy(self, y_true, y_pred):


    return -np.mean(y_true*np.log(y_pred)+(1-y_true)*np.log(1-y_pred))


  def gradient_cross_entropy(self, X,y_true,y_pred):


    return (-1/X.shape[0])*X.T@(y_true-y_pred)

  def add_ones(self,x):

    ones = np.ones(x.shape[0]).reshape(-1,1)
    return np.hstack((ones,x))

  def proba_pred(self,x,theta):
    z = self.linear(x,theta)


    return 1/(1+np.exp(-z))



  def Initialization(self,D):

    return np.zeros(D).reshape(-1,1)

  def update(self, theta, lr, grad):

    return theta- lr*grad

  def fit(self, X,y, lr=0.01,n_epoch = 1000):

    n,d = X.shape
    self.theta = self.Initialization(d)


    for epoch in range(n_epoch):


      y_pred = self.proba_pred(X,self.theta)


      loss = self.cross_entropy(y,y_pred)

      self.losses.append(loss)

      if epoch%100 ==0:
        print(f" at epoch = {epoch}, loss = {loss}")

      grad = self.gradient_cross_entropy(X,y,y_pred)

      if np.linalg.norm(grad)<0.005:
        break

      self.theta = self.update(self.theta,lr,grad)


  def predict(self, X):

    proba = self.proba_pred(X,self.theta)

    return (proba > 0.5).astype(int)

  def  accuracy(self, y_true,y_pred):

    return np.sum(y_true==y_pred)/y_true.shape[0]



if __name__ == "__main__":

  Logistic_Regression()
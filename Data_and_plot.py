
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




def split_data(x,y,train_perc=0.8):
  N=x.shape[0]
  train_size=round(train_perc*N)
  x_train,y_train=x[:train_size,:],y[:train_size]
  x_test,y_test=x[train_size:,:],y[train_size:]
  return x_train,y_train,x_test,y_test


def plot_loss(loss):
    plt.plot(loss)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

def plot(x_test,y_test,y_pred):
  
    plt.scatter(x_test,y_test, label = " data test")
    plt.plot(x_test,y_pred,color = "red", label = " model")

    plt.legend()

    plt.show()

if __name__ == "__main__":
      generate_data()

      split_data()

      plot_loss()

      plot()
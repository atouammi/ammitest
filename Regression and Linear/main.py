from Data_and_plot import  generate_data, split_data,plot_loss,plot
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from LinearRegression_withSimpleGD import Linear_regression
from LinearRegression_StochasticGD import LinearRegressionSGD
from LinearRegression_with_minibatchGD import LinearRegressionMb
from LinearRegression_withSGDmomentum import LinearRegressionMm
from LogisticReg import Logistic_Regression









X_class, y_class = make_classification(n_samples= 100, n_features=2, n_redundant=0,
                           n_informative=2, random_state=1,
                           n_clusters_per_class=1)



# x,y =generate_data()


x_train,y_train,x_test,y_test=split_data(X_class,y_class)
print(f"x_train:{x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}")



plt.scatter(x_train[:,0],x_train[:,1], c=y_train)
plt.xlabel("feature")
plt.ylabel('target')
plt.show()

X,y = generate_data()





# model = Linear_regression()

# model = LinearRegressionSGD()

# model = LinearRegressionMb()

#model = LinearRegressionMm()

model  = Logistic_Regression()


model.fit(x_train,y_train)


plot_loss(model.losses)





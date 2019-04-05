import pandas as pd
poker = pd.read_csv('D:\poker-hand-training-true.data', names = ["1s", "1v", "2s", "2v", "3s", "3v", "4s", "4v", "5s", "5v", "Hand"])
x_train = poker.drop('Hand',axis=1)
y_train = poker['Hand']

poker_test = pd.read_csv('D:\poker-hand-testing.data', names = ["1s", "1v", "2s", "2v", "3s", "3v", "4s", "4v", "5s", "5v", "Hand"])
x_test = poker_test.drop('Hand',axis=1)
y_test = poker_test['Hand']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(x_train)
StandardScaler(copy=True, with_mean=True, with_std=True)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(38,38,38),max_iter=2500,alpha=0.0001,activation='relu', learning_rate_init=0.009)
mlp.fit(x_train,y_train)

predictions = mlp.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


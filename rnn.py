import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("tr.csv")
training_set = data[["Open"]]

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_sc = sc.fit_transform(training_set)

x_train = []
y_train = []

for i in range(60, 1258):
    x_train.append(training_set_sc[i-60:i, 0])
    y_train.append(training_set_sc[i, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

# RNN (Rows=1198 , Columns=60, Features=1)
x_train = np.reshape(x_train, (1198, 60, 1))

# Create RNN
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(40, return_sequences=True ,input_shape=(60,1)))
model.add(Dropout(0.2))
model.add(LSTM(40, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(40))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")

model.fit(x_train, y_train, epochs=120, batch_size=32)

data_test = pd.read_csv("ts.csv")
real_prices = data_test[["Open"]]

dataset_total = pd.concat((data["Open"], data_test["Open"]), axis=0)
inputs = dataset_total[len(dataset_total) - len(data_test) - 60:].values

print(inputs.shape)

inputs = inputs.reshape(-1, 1)
print(inputs.shape)

inputs = sc.transform(inputs)

x_test = []
for i in range(60, 80):
    x_test.append(inputs[i-60:i, 0])

x_test = np.array(x_test)
# RNN (Rows=20 , Columns=60, Features=1)
x_test = np.reshape(x_test, (20, 60, 1))

#Prediction
pred_prices = model.predict(x_test)
pred_prices = sc.inverse_transform(pred_prices)

plt.plot(real_prices, c="blue")
plt.plot(pred_prices, c="red")
plt.show()
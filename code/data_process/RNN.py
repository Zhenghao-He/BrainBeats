import numpy as np
from preprocess import *
from keras.utils import np_utils
from keras.models import Sequential  # 构建网络必需模块
from keras.layers import SimpleRNN, Activation, Dense  # RNN、激活函数、全连接层模块
from keras.optimizers import Adam  # 优化器模块
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from sklearn.metrics import log_loss
from keras import metrics
import wandb
from wandb.keras import WandbMetricsLogger

wandb.init(config={"bs": 12})
print(x_close.shape)
x_close = x_close.T.reshape(1200, 1, 15)
y_close = to_categorical(y_close)
X_train, X_test, y_train, y_test = train_test_split(x_close, y_close, test_size=0.2, random_state=42)

model = Sequential()
model.add(SimpleRNN(units=32, input_shape=(1, 15)))
model.add(Dense(units=2, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',metrics.mean_squared_error, metrics.mean_absolute_error])
model.fit(x_close, y_close, epochs=60, batch_size=32)
# x_close = x_close.reshape((1200, 15, 1))
# y_close = to_categorical(y_close)
# 训练模型
# X_train, X_test, y_train, y_test = train_test_split(x_close, y_close, test_size=0.2, random_state=42)
model.fit(x_close, y_close, epochs=100, batch_size=32,callbacks=[WandbMetricsLogger()])
y_hat=model.predict(X_test)

loss = log_loss(y_test, y_hat)
print(loss)
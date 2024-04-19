from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import log_loss
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from preprocess import *
import wandb
from wandb.keras import WandbMetricsLogger
from keras import metrics
# 构建LSTM模型
wandb.init(project="uncategorized",name='LSTM_1')
model = Sequential()
model.add(LSTM(units=32, input_shape=(15, 1)))
model.add(Dense(units=2, activation='sigmoid'))

# 编译模型
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',metrics.mean_squared_error, metrics.mean_absolute_error])

# 将输入数据reshape为 (1200, 15, 1)，其中1代表通道数
x_close = x_close.reshape((1200, 15, 1))
y_close = to_categorical(y_close)
# 训练模型
X_train, X_test, y_train, y_test = train_test_split(x_close, y_close, test_size=0.2, random_state=42)
model.fit(x_close, y_close, epochs=100, batch_size=32,callbacks=[WandbMetricsLogger()])
y_hat=model.predict(X_test)

loss = log_loss(y_test, y_hat)

# 输出交叉熵损失
print(loss)
wandb.close()
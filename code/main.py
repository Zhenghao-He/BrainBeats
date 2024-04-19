import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from conformer import Conformer
import scipy.io as scio
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from matplotlib import pyplot as plt
from Preprocess import EEG2Hz_data, EEG2Hz_label, fdata


def plot_history(history, name):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 10000])
    plt.legend()
    plt.savefig('./pics/' + name + '_mae.png')  # 保存为PNG格式
    plt.clf()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 20000])
    plt.legend()
    # plt.show()
    plt.savefig('./pics/' + name + '_mse.png')  # 保存为PNG格式
    plt.clf()


for data, label, fname in zip(EEG2Hz_data, EEG2Hz_label, fdata):
    X = data['psd_movingAve']
    X = X.transpose(1, 0, 2)
    X = np.expand_dims(X, axis=3)
    # train_inputs = X[0:700, :, :]
    train_inputs = X[0:700, :, :, :]
    valid_inputs = X[700:885, :, :, :]
    # valid_inputs = X[700:885, :, :]
    train_labels = label['perclos'][0:700]
    valid_labels = label['perclos'][700:885]

    # 将列表转换为张量
    train_inputs = torch.tensor(train_inputs).float()
    train_labels = torch.tensor(train_labels).float()
    valid_inputs = torch.tensor(valid_inputs).float()
    valid_labels = torch.tensor(valid_labels).float()
    # 创建 TensorDataset
    train_data = TensorDataset(train_inputs, train_labels)
    valid_data = TensorDataset(valid_inputs, valid_labels)
    # 创建 DataLoader
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    # 定义模型
    model = Conformer(emb_size=40, depth=6, n_classes=4)

    # 定义损失函数
    criterion = nn.MSELoss()

    # 定义优化器
    # optimizer = optim.SGD(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # 训练循环
    num_epochs = 100
    fname = fname.split(".")[0]
    print(fname)
    mse_values = []
    mae_values = []
    mae_variance = []
    mse_variance = []
    for epoch in range(num_epochs):
        model.train()
        # 初始化变量用于累计MAE和MSE
        total_mae = 0
        total_mse = 0
        for inputs, labels in train_loader:
            # 前向传播
            x, outputs = model(inputs)

            # 计算损失
            # labels = torch.tensor(labels)
            # print(outputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 计算MAE和MSE
            mae = nn.L1Loss()(outputs, labels)
            mse = loss
            total_mae += mae.item() * labels.size(0)
            total_mse += mse.item() * labels.size(0)

        # 在验证集上评估模型
        model.eval()
        total_loss = 0
        total_samples = 0



        with torch.no_grad():
            for inputs, labels in valid_loader:
                x, outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)

        avg_mae = total_mae / len(train_loader)
        avg_mse = total_mse / len(train_loader)
        # 记录MSE和MAE的值
        mse_values.append(avg_mse)
        mae_values.append(avg_mae)

        rmse = torch.sqrt(torch.tensor(total_loss) / total_samples)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation RMSE: {rmse}, MAE: {avg_mae}, MSE: {avg_mse}")

    # 绘制MSE和MAE的变化曲线
    epochs = range(1, num_epochs + 1)

    plt.plot(epochs, mse_values, label='MSE')
    plt.plot(epochs, mae_values, label='MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('MSE and MAE During Training')
    plt.legend()
    # plt.show()
    plt.savefig('./pics/' + fname + '_mae&mse.png')  # 保存为PNG格式
    plt.clf()

    # 在验证集上进行预测
    with torch.no_grad():
        model.eval()
        x, outputs = model(valid_inputs)

    Y_pred = outputs  # 假设输出即为预测结果

    # 计算MSE
    mse = torch.mean((Y_pred - valid_labels) ** 2)
    mse_values.append(mse)

    # 计算MAE
    mae = torch.mean(np.abs(Y_pred - valid_labels))
    mae_values.append(mae)

    # 计算方差
    mae_var = torch.var(torch.abs(Y_pred - valid_labels))
    mse_var = torch.var((Y_pred - valid_labels) ** 2)
    mae_variance.append(mae_var)
    mse_variance.append(mse_var)
    print("file_name:", fname)
    print("mse:", mse)
    print("mae:", mae)
    print("mse_var:", mse_var)
    print("mae_var:", mae_var)
    # 训练完成后，你可以保存模型的参数
    torch.save(model.state_dict(), './models/conformer_model' + fname + '.pth')

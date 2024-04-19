"""
说明：
《脑科学导论》课程2023春季	普适性脑机接口
实验二、脑机接口数据采集与特征分析验证性实验
采集数据设备：NeuroSky 的TGAM
脑认知任务设计：8s的实验引导语，0：08-3：08静息状态3分钟，3：09-3：13指导语，3：14-6：12听音乐3分钟，最后56s指导语

特征分析验证目标：

Part1：脑电信号读入与预处理
1. 从原始信号中截取信号：0：40-2：40保存为REST状态, 4min-6min保存为LISTEN状态
2. 对截取信号进行分段：5s为一段，进行数据扩充
3. 基于5s信号进行数据平均后，画出RESTvsLISTEN的原始信号对比图

Part2：脑电信号特征提取
1. 根据文献进行脑电信号的15个特征的计算
2. 画出RESTvsLISTEN的15个特征的分布对比图

Part3：脑电信号特征分类
1. 利用SVM作为分类器，SP15特征为输入信号，进行REST vs LISTEN的分类任务
2. 在分析过程中，需要记录及输入分类准确率和实验时间。
3. 请使用英文的文件路径！！！！
"""


'''
	库函数导入
'''
import itertools
import os 
import scipy
import numpy
from scipy import signal 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedKFold


"""
	脑电信号数据路径
"""

DATA_PATH = [
	# "/content/drive/MyDrive/BrainLink/BrainLink2/BrainLinkRaw_01_FAT.txt",
	# "/content/drive/MyDrive/BrainLink/BrainLink2/BrainLinkRaw_03_FAT2.txt",
	# "/content/drive/MyDrive/BrainLink/BrainLink2/BrainLinkRaw_04_FAT.txt",
	# "/content/drive/MyDrive/BrainLink/BrainLink2/BrainLinkRaw_05_FAT.txt"
    "H:\EEG-design\FAT-WAKE\BrainLink\data\BrainLinkRaw_01_FAT.txt",
    "H:\EEG-design\FAT-WAKE\BrainLink\data\BrainLinkRaw_02_FAT.txt",
    "H:\EEG-design\FAT-WAKE\BrainLink\data\BrainLinkRaw_01_WAK.txt",
    "H:\EEG-design\FAT-WAKE\BrainLink\data\BrainLinkRaw_02_WAK.txt"
	
	
]

'''
	定义常量
'''

# REST 5s时间窗起始点
rest_start = 8
# REST 5s时间窗终止点
rest_end = 32
# LISTEN 5s时间窗起始点
listen_start = 48
# LISTEN 5s时间窗终止点
listen_end = 72
# TGAM设备的采样率为512，即1s中存储512个数据点，记录数据的单位为uv
sample_rate = 512

"""
	A Norch Filter:
	Band-stop filter
	stop frequency: 50Hz
"""
def Notch_filter(sig_data):

  fr = 50
  fs = sample_rate
  Q = 10
  b,a = signal.iirnotch(fr,Q,fs)
  filted_data=scipy.signal.filtfilt(b,a,sig_data)
  # 配置滤波器 8 表示滤波器的阶数
  b, a = signal.butter(8, [0.004,0.86], 'bandpass')
  # data为要过滤的信号
  filtedData = signal.filtfilt(b, a, filted_data)

  return filtedData


"""FFT with hamming window"""
def FFT_ham(sig_data):
  sample_rate=512
  N=5*sample_rate
  
  hamming_win=signal.hamming(N)
  sig_win=sig_data*hamming_win

  sig_fft=numpy.fft.fft(sig_win)

  f=numpy.fft.fftfreq(N, 1/sample_rate)

  #plt.plot(f, np.abs(sig_fft))
  #plt.xlabel('Freq')
  #plt.ylabel('Magnitude')
  #plt.show()
  return sig_fft, f


"""
	Part1:脑电数据读入，数据分断、滤波及预处理
"""

# 分类标签，在TGAM测试实验中，REST 状态KSS=0，LISTEN状态 KSS=1
KSS=[]
index = 0
delta, theta, alpha, beta, gamma, fi, KSS = [], [], [], [], [], [], []
filtered_data=numpy.zeros(sample_rate*5)

# REST 数据处理
for file in DATA_PATH:
	filtered_data=numpy.zeros(sample_rate*5)
	EEG = numpy.loadtxt(file)
	for i in range(rest_start, rest_end):
	  filtered_data=filtered_data+Notch_filter((EEG[i*sample_rate*5:(i+1)*sample_rate*5]))
	  sig_fft,freq_fft=FFT_ham(filtered_data)
	  eeg_power=numpy.abs(sig_fft)**2
	  eeg=eeg_power[:int(sample_rate*5/2)]

	  # 数据处理
	  delta.append(numpy.mean(eeg[int(0.5/128*(sample_rate*5/2)):int(4/128*(sample_rate*5/2))]))
	  theta.append(numpy.mean(eeg[int(4/128*(sample_rate*5/2)):int(8/128*(sample_rate*5/2))]))
	  alpha.append(numpy.mean(eeg[int(7.5/128*(sample_rate*5/2)):int(13/128*(sample_rate*5/2))]))
	  beta.append(numpy.mean(eeg[int(13/128*(sample_rate*5/2)):int(30/128*(sample_rate*5/2))]))
	  gamma.append(numpy.mean(eeg[int(30/128*(sample_rate*5/2)):int(44/128*(sample_rate*5/2))]))
	  fi.append(numpy.mean(eeg[int(0.85/128*(sample_rate*5/2)):int(110/128*(sample_rate*5/2))]))

	  # KSS.append(KSS_val)
	  KSS.append(0)

# REST平均值
relaxave=filtered_data/4/24

# LISTEN 数据处理
filtered_data=numpy.zeros(sample_rate*5)
for file in DATA_PATH:
	filtered_data=numpy.zeros(sample_rate*5)
	EEG = numpy.loadtxt(file)
	for i in range(listen_start, listen_end):
	  filtered_data=filtered_data+Notch_filter((EEG[i*sample_rate*5:(i+1)*sample_rate*5]))  
	  sig_fft,freq_fft=FFT_ham(filtered_data)
	  eeg_power=numpy.abs(sig_fft)**2
	  eeg=eeg_power[:int(sample_rate*5/2)]
	  
	  # 数据处理
	  delta.append(numpy.mean(eeg[int(0.5/128*(sample_rate*5/2)):int(4/128*(sample_rate*5/2))]))
	  theta.append(numpy.mean(eeg[int(4/128*(sample_rate*5/2)):int(8/128*(sample_rate*5/2))]))
	  alpha.append(numpy.mean(eeg[int(7.5/128*(sample_rate*5/2)):int(13/128*(sample_rate*5/2))]))
	  beta.append(numpy.mean(eeg[int(13/128*(sample_rate*5/2)):int(30/128*(sample_rate*5/2))]))
	  gamma.append(numpy.mean(eeg[int(30/128*(sample_rate*5/2)):int(44/128*(sample_rate*5/2))]))
	  fi.append(numpy.mean(eeg[int(0.85/128*(sample_rate*5/2)):int(110/128*(sample_rate*5/2))]))
	  KSS.append(1)

# LISTEN平均值
musicave=filtered_data/4/24


"""
	Part2:脑电信号基本特征计算
	计算15个特征
"""

delta = numpy.array(delta)  
theta = numpy.array(theta)    
alpha = numpy.array(alpha)    
beta = numpy.array(beta)    
gamma = numpy.array(gamma)    
fi = numpy.array(fi)   
 
Sp = [delta, theta, alpha, theta/beta, theta/alpha, theta/fi, theta/alpha+beta+gamma, delta/alpha+beta+gamma, delta/alpha, delta/fi, \
	  delta/beta, delta/theta, theta/alpha+beta+theta, alpha/theta+alpha+beta, beta/theta+alpha+beta]


"""
	Part3：认知任务分类
	利用SVM作为分类器，SP15特征为输入信号，进行REST vs LISTEN的分类任务
"""
#  classify 
x = numpy.transpose(numpy.array(Sp))
y = numpy.transpose(numpy.array(KSS))
print("Sp Shape：",x.shape)
print("KSS Shape:",y.shape)

# x_open=x[:192]
# y_open=y[:192]

# Open eyes classify 
x = numpy.transpose(numpy.array(Sp))
y = numpy.transpose(numpy.array(KSS))


average_acc = 0.0
kf = RepeatedKFold(n_splits=10, n_repeats=2)
acc = numpy.zeros(20)
index = 0
for train_index, test_index in kf.split(x):
 train_X= x[train_index]
 train_y =y[train_index]
 test_X, test_y = x[test_index], y[test_index]
 regr = make_pipeline(StandardScaler(), SVC())
 regr.fit(train_X, train_y)
 y_pred = regr.predict(test_X)
 #print(accuracy_score(test_y, y_pred))
 acc[index] = accuracy_score(test_y, y_pred)
 index = index + 1
print("BrainLink MIX average_acc:", numpy.mean(acc), "std_acc:", 
numpy.std(acc))


"""PART4: 结果可视化输出1
# Plot result
* 5s waveform with normalized amplitude
"""

def normalize(x):
  arange=x.max()-x.min()
  y=(x-x.min())/arange-(x.mean()-x.min())/arange
  return y


plt.rcParams.update({'font.size': 20}) 
x=numpy.arange(0,5,(5/2560))

plt.figure(figsize=(20,5))
plt.title("TGAM")
plt.plot(x,normalize(relaxave),label='RELAX')
plt.plot(x,normalize(musicave),label='MUSIC')
plt.xlabel("Time(s)")
plt.ylabel("EEG")
plt.ylim(-1,1)
plt.legend()


"""PART4: 结果可视化输出2
# Plot result
* 15 Sp features distribution
"""
xlabel=['delta','theta','alpha','theta/beta','theta/alpha','theta/fi','theta/alpha+beta+gamma',
		'delta/alpha+beta+gamma','delta/alpha','delta/fi','delta/beta','delta/theta','theta/alpha+beta+theta',
		'alpha/theta+alpha+beta','beta/theta+alpha+beta']
for i in range(0,15):
  
  plt.figure(figsize=(20,5))
  # plt.subplot(121)
  # plt.hist(SPWK[i], 432)
  # plt.subplot(122)
  # plt.hist(SPSL[i], 432)
  plt.title("TGAM Sp"+str(i+1))
  plt.hist([Sp[i][0:96],Sp[i][96:192]],100,histtype='bar',label=['RELAX','MUSIC'])
  plt.ylabel("Probability")
  plt.xlabel(xlabel[i])
  # ax1.set_ylim(0,1)
  # print(Sp[i][0:96])
  # ax1.hist(WKO[i],100,label='WAKE-OPEN')
  plt.legend()

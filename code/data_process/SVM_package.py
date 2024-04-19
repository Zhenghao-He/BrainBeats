from preprocess import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import RepeatedKFold
import wandb


average_acc = 0.0
kf = RepeatedKFold(n_splits=10, n_repeats=2)
acc = numpy.zeros(20)
index = 0
for train_index, test_index in kf.split(x_open):
    #wandb.init(project="uncategorized",name='SVM_open_index' + str(index))
    train_X = x_open[train_index]
    train_y = y_open[train_index]
    test_X, test_y = x_open[test_index], y_open[test_index]
    regr = make_pipeline(StandardScaler(), SVC(probability=True))
    regr.fit(train_X, train_y)
    y_pred = regr.predict(test_X)
    # print(accuracy_score(test_y, y_pred))
    acc[index] = accuracy_score(test_y, y_pred)
    index = index + 1
    y_probas = regr.predict_proba(test_X)
    #wandb.sklearn.plot_classifier(regr, train_X, test_X, train_y, test_y, y_pred,y_probas,['FAT','WAK'],
    #                                                    model_name='SVC', feature_names=None)
    #wandb.finish()
print("BrainLink Open Eyes average_acc:", numpy.mean(acc), "std_acc:",
      numpy.std(acc))

average_acc = 0.0
kf = RepeatedKFold(n_splits=10, n_repeats=2)
acc = numpy.zeros(20)
index = 0
for train_index, test_index in kf.split(x_close):
    #wandb.init(project="uncategorized",name='SVM_close_index' + str(index))
    train_X = x_close[train_index]
    train_y = y_close[train_index]
    test_X, test_y = x_close[test_index], y_close[test_index]
    regr = make_pipeline(StandardScaler(), SVC(probability=True))
    regr.fit(train_X, train_y)
    y_pred = regr.predict(test_X)
    # print(accuracy_score(test_y, y_pred))
    acc[index] = accuracy_score(test_y, y_pred)
    index = index + 1
    y_probas = regr.predict_proba(test_X)
    #wandb.sklearn.plot_classifier(regr, train_X, test_X, train_y, test_y, y_pred,y_probas,['FAT','WAK'],
    #                                                    model_name='SVC', feature_names=None)
    #wandb.finish()
print("BrainLink Close Eyes average_acc:", numpy.mean(acc), "std_acc:",
      numpy.std(acc))

average_acc = 0.0
kf = RepeatedKFold(n_splits=10, n_repeats=2)
acc = numpy.zeros(20)
index = 0
for train_index, test_index in kf.split(x):
    #wandb.init(project="uncategorized",name='SVM_MIX_index' + str(index))
    train_X = x[train_index]
    train_y = y[train_index]
    test_X, test_y = x[test_index], y[test_index]
    regr = make_pipeline(StandardScaler(), SVC(probability=True))
    regr.fit(train_X, train_y)
    y_pred = regr.predict(test_X)
    # print(accuracy_score(test_y, y_pred))
    acc[index] = accuracy_score(test_y, y_pred)
    index = index + 1
    y_probas = regr.predict_proba(test_X)
    #wandb.sklearn.plot_classifier(regr, train_X, test_X, train_y, test_y, y_pred,y_probas,['FAT','WAK'],
    #                                                    model_name='SVC', feature_names=None)
    #wandb.finish()
print("BrainLink MIX average_acc:", numpy.mean(acc), "std_acc:",
      numpy.std(acc))
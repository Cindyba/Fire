import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score


def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def R2(pred, true):
    pred = pred.reshape((1, pred.shape[0]), order='A')
    pred = pred[0, :]
    true = true.reshape((1, true.shape[0]), order='A')
    true = true[0, :]
    return r2_score(true, pred)
    #sum((pred-true)**2) / sum((true - np.mean(true))**2)

def ACC(pred, true):
    acc = 0
    for i in range(len(pred)):
        if (pred[i][0][0] < 0.5 and true[i][0][0] == 0 ) or (pred[i][0][0] >= 0.5 and true[i][0][0] == 1):
            acc += 1
    acc /= len(pred)
    return acc

def Precision(pred, true):
    TP = 0
    P_ = 0
    TN = 0
    N_ = 0
    for i in range(len(pred)):
        if pred[i][0][0] >= 0.5:
            P_ += 1
        else:
            N_ += 1
        if pred[i][0][0] >= 0.5 and true[i][0][0] == 1 :
            TP += 1
        elif pred[i][0][0] < 0.5 and true[i][0][0] == 0 :
            TN += 1
    return TP/P_, TN/N_

def Recall(pred, true):
    P = 0
    TP = 0
    N = 0
    TN = 0
    for i in range(len(pred)):
        if true[i][0][0] == 1:
            P += 1
        else:
            N += 1
        if pred[i][0][0] >= 0.5 and true[i][0][0] == 1 :
            TP += 1
        elif pred[i][0][0] < 0.5 and true[i][0][0] == 0 :
            TN += 1
    return TP/P, TN/N

def AUC(pred, true):
    pred = pred.reshape((1, pred.shape[0]), order='A')
    pred = pred[0, :]
    true = true.reshape((1, true.shape[0]), order='A')
    true = true[0, :]
    return roc_auc_score(true, pred)

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = R2(pred, true)
    '''acc = ACC(pred, true)
    precisionT, precisionN = Precision(pred, true)
    recallT, recallN = Recall(pred, true)
    auc = AUC(pred, true)'''

    #return acc, precisionT, precisionN, recallT, recallN, auc
    return mae,mse,rmse,mape,mspe,r2
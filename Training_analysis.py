import matplotlib.pyplot as plt
import csv
import re
import torch

def load_log(path):
    train_loss = []
    test_loss = []
    vali_loss = []
    with open(path, 'r') as file:
        for line in file.readlines():
            line = line.strip('\n')  # 去掉列表中每一个元素的换行符
            print(line)
            train_pattern = re.compile(r'Train Loss: (0.\d+)')
            trLoss = train_pattern.findall(line)
            test_pattern = re.compile(r'Test Loss: (0.\d+)')
            teLoss = test_pattern.findall(line)
            vali_pattern = re.compile(r'Vali Loss: (0.\d+)')
            vLoss = vali_pattern.findall(line)
            print(trLoss)
            if len(trLoss) != 0:
                train_loss.append(float(trLoss[0]))
                test_loss.append(float(teLoss[0]))
                vali_loss.append(float(vLoss[0]))
            if line == "Early stopping" :
                break
    print(train_loss)
    print(test_loss)
    print(vali_loss)
    return train_loss, test_loss, vali_loss


def show(y, y1, y2):
    x = range(len(y))
    print(x)
    print(y1)

    plt.plot(x, y, marker='o', mec='r', mfc='w', label=u'train_loss')
    plt.plot(x, y1, marker='*', ms=10, label=u'test_loss')
    plt.plot(x, y2, marker='^', ms=10, label=u'vali_loss')
    plt.legend()  # 让图例生效
    plt.margins(0)
    plt.subplots_adjust(bottom=0.15)
    plt.xlabel(u"epoch")  # X轴标签
    plt.ylabel("Loss")  # Y轴标签

    plt.show()



train_loss, test_Loss, vali_loss = load_log(path='log_img.txt')
show(train_loss, test_Loss, vali_loss)

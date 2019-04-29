import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)
path = 'G://'
sys.path.append(path)

import numpy as np
import matplotlib.pyplot as plt
import math
from classification_utils import *
from regression_utils import *

#classification loss functions
def log_loss(y_true, y_pred):
    return y_true * np.log2(1 + np.exp(-y_pred)) - (1-y_true) * np.log2(1 - 1/(1+np.exp(-y_pred)))

def hinge_loss(y_true, y_pred):
    return np.where(y_true*y_pred < 1, 1 - y_true * y_pred, 0)

def squared_hinge_loss(y_true, y_pred):
    return np.where(y_true*y_pred < 1, 1 - y_true * y_pred, 0) ** 2

def perceptron_loss(y_true, y_pred):
    return -np.minimum(y_true * y_pred, 0) 

def modified_huber_loss(y_true, y_pred):
    z = y_pred * y_true
    loss = -4 * z
    loss[z >= -1] = (1 - z[z >= -1]) ** 2
    loss[z >= 1.] = 0
    return loss

def plot_classification_loss_single_model_combined(decision_min, decision_max):        
    y_pred = np.linspace(decision_min, decision_max, 100)

    plt.plot([decision_min, 0, 0, decision_max], [1, 1, 0, 0], color='yellow', lw=2, label="Zero-one loss")

    plt.plot(y_pred, hinge_loss(1, y_pred), color='teal', lw=2, label="Hinge loss")
    plt.plot(y_pred, hinge_loss(-1, y_pred), color='teal', lw=2, label="Hinge loss")

    plt.plot(y_pred, squared_hinge_loss(1, y_pred), color='orange', lw=2, label="Squared hinge loss")
    plt.plot(y_pred, squared_hinge_loss(-1, y_pred), color='orange', lw=2, label="Squared hinge loss")

    plt.plot(y_pred, perceptron_loss(1, y_pred), color='yellowgreen', lw=2, label="Perceptron loss")
    plt.plot(y_pred, perceptron_loss(-1, y_pred), color='yellowgreen', lw=2, label="Perceptron loss")

    plt.plot(y_pred, log_loss(1, y_pred), color='cornflowerblue', lw=2, label="Log loss")
    plt.plot(y_pred, log_loss(0, y_pred), color='cornflowerblue', lw=2, label="Log loss")

    plt.plot(y_pred, modified_huber_loss(1, y_pred), color='darkorchid', lw=2, label="Modified Huber loss")
    plt.plot(y_pred, modified_huber_loss(-1, y_pred), color='darkorchid', lw=2, label="Modified Huber loss")

    plt.ylim((0, 8))
    plt.xlabel("Decision function $wx$")
    plt.ylabel("$Loss(y, wx)$")
    plt.legend(loc="upper right")
    plt.show()

def plot_classification_loss_single_model(X, y, m, b, titles):
    losses = []    
    decisions = X[:,1] - m * X[:,0] - b    
    losses.append(np.round(decisions,2))    
    y_trans = y.copy()
    y_trans[y_trans==0] = -1
    losses.append(np.round(hinge_loss(y_trans, decisions),2))
    losses.append(np.round(squared_hinge_loss(y_trans, decisions),2))
    losses.append(np.round(perceptron_loss(y_trans, decisions),2))
    losses.append(np.round(log_loss(y, decisions),2))
    losses.append(np.round(modified_huber_loss(y_trans, decisions),2)) 

    xfit = np.linspace(-2, 2)

    plt.style.use('seaborn')
    colors = "ryb"
    n_classes = set(y)
    class_labels = [str(i) for i in n_classes]
    fig, axes = plt.subplots(2, 3, figsize=(12,12))    
    for ax, loss, title in zip(axes.reshape(-1), losses, titles):
        for i, color in zip(n_classes, colors):
            idx = np.where(y == i)
            ax.scatter(X[idx, 0], X[idx, 1], c=color, label = class_labels[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=30) 
        ax.plot(xfit, m * xfit + b, '-k')
        for i, txt in enumerate(loss):
            ax.annotate(txt, (X[i,0], X[i,1]))
        ax.set_xlim((-2, 2))
        ax.set_ylim((-2, 2))
        ax.legend()
        ax.set_title(title)

#generate synthetic data for binary classification
X, y = generate_linear_synthetic_data_classification(50, 2, 2, [.5,.5])
plot_data_2d_classification(X, y)
y1 = y.copy()
y1[y==0] = 1
y1[y==1] = 0

classf_loss_titles = ['Distances from Line', 'Hinge Loss', 'Squared Hinge Loss', 'Perceptron Loss', 'Log Loss', 'Modified Huber Loss']
plot_classification_loss_single_model(X, y1, 1, 0.65, classf_loss_titles)
plot_classification_loss_single_model_combined(-4, 4)
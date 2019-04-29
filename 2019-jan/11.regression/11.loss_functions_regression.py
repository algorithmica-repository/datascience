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

#regression loss functions
def squared_loss(y_true, y_pred):
    return (y_true - y_pred) ** 2

def absolute_loss(y_true, y_pred):
    return np.abs(y_true - y_pred)

def huber_loss(y_true, y_pred, delta):
    return np.where(np.abs(y_true-y_pred) < delta , 0.5*((y_true-y_pred)**2), delta*np.abs(y_true - y_pred) - 0.5*(delta**2))

def epsilon_insensitive_loss(y_true, y_pred, epsilon):  
    return np.where(np.abs(y_true - y_pred) <= epsilon, 0.0, np.abs(y_true - y_pred) - epsilon)

def plot_regression_loss_single_model(X, y, m, b, titles):
    losses = []    
    y_pred = m * X[:,0] + b 
    losses.append(np.round(y,2))    
    losses.append(np.round(squared_loss(y, y_pred),2))
    losses.append(np.round(absolute_loss(y, y_pred),2))
    losses.append(np.round(huber_loss(y, y_pred, 0.2),2))
    losses.append(np.round(huber_loss(y, y_pred, 5.0),2))
    losses.append(np.round(epsilon_insensitive_loss(y, y_pred, 1.0),2))
    
    xfit = np.linspace(-3, 3)

    plt.style.use('seaborn')
    fig, axes = plt.subplots(2, 3, figsize=(12,12))    
    for ax, loss, title in zip(axes.reshape(-1), losses, titles):
        ax.scatter(X[:, 0], y, c='blue', cmap=plt.cm.RdYlBu, edgecolor='black', s=30) 
        ax.plot(xfit, m * xfit + b, '-k')
        for i, txt in enumerate(loss):
            ax.annotate(txt, (X[i,0], y[i]))
        ax.legend()
        ax.set_title(title)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-40,40)

def plot_regression_loss_single_model_combined(decision_min, decision_max):        
    y_pred = np.linspace(decision_min, decision_max, 100)

    plt.plot(y_pred, squared_loss(0, y_pred), color='teal', lw=2, label="Squared loss")
    plt.plot(y_pred, absolute_loss(0, y_pred), color='yellowgreen', lw=2, label="Absolute loss")
    plt.plot(y_pred, huber_loss(0, y_pred, 0.2), color='orange', lw=2, label="Huber loss(0.2)")
    plt.plot(y_pred, huber_loss(0, y_pred, 5.0), color='orange', lw=2, label="Huber loss(5.0)")
    plt.plot(y_pred, epsilon_insensitive_loss(0, y_pred, 0.0), color='darkorchid', lw=2, label="Epsilon Insensitive Loss(0.0)")
    plt.plot(y_pred, epsilon_insensitive_loss(0, y_pred, 1.0), color='darkorchid', lw=2, label="Epsilon Insensitive Loss(1.0)")

    plt.ylim((0, 25))
    plt.xlabel("Decision function $wx$")
    plt.ylabel("$Loss(y, wx)$")
    plt.legend(loc="upper right")
    plt.show()
    
#generate synthetic data for regression
X, y = generate_linear_synthetic_data_regression(20, 1, 1, 15)
plot_data_2d_regression(X, y)

reg_loss_titles = ['Distances from Line', 'Squared Loss', 'Absolute Loss', 'Huber Loss(0.2)', 'Huber Loss(5.0)', 'Epsilon Insensitive Loss(1.0)']
plot_regression_loss_single_model(X, y, 4, 1, reg_loss_titles)
plot_regression_loss_single_model_combined(-10, 10)

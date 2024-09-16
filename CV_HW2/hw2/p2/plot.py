import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from model import MyNet, ResNet18
from dataset import get_dataloader
from utils import set_seed, write_config_log, write_result_log

import config as cfg

def plot_learning_curve(logfile_dir, result_lists):
    ################################################################
    # TODO:                                                        #
    # Plot and save the learning curves under logfile_dir, you can #
    # use plt.plot() and plt.savefig().                            #
    #                                                              #
    # NOTE:                                                        #
    # You have to attach four plots of your best model in your     #
    # report, which are:                                           #
    #   1. training accuracy                                       #
    #   2. training loss                                           #
    #   3. validation accuracy                                     #
    #   4. validation loss                                         #
    #                                                              #
    # NOTE:                                                        #
    # This function is called at end of each epoch to avoid the    #
    # plot being unsaved if early stop, so the result_lists's size #
    # is not fixed.                                                #
    ################################################################
    plt.figure(figsize=(10, 6))
    plt.plot(result_lists['train_acc'], label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(logfile_dir, 'training_accuracy_curve.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(result_lists['train_loss'], label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(logfile_dir, 'training_loss_curve.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(result_lists['val_acc'], label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(logfile_dir, 'validation_accuracy_curve.png'))
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(result_lists['val_loss'], label='Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(logfile_dir, 'validation_loss_curve.png'))
    plt.close()
    pass

def main():
    
    logfile_dir = os.path.join('./experiment', 'mynet_2024_04_16_15_23_06_sgd_pre_da', 'log')
    
    with open (os.path.join(logfile_dir, 'result_log.txt'), 'r') as f:
        data = f.readlines()
    
    train_acc_list = []
    train_loss_list = []
    val_acc_list = []
    val_loss_list = []
    
    for i in range(len(data)):
        data[i] = data[i].strip().split()
        train_acc_list.append(float(data[i][5]))
        train_loss_list.append(float(data[i][13]))
        val_acc_list.append(float(data[i][9]))
        val_loss_list.append(float(data[i][17]))
    
    current_result_lists = {
        'train_acc': train_acc_list,
        'train_loss': train_loss_list,
        'val_acc': val_acc_list,
        'val_loss': val_loss_list
    }
    
    plot_learning_curve(logfile_dir, current_result_lists)

if __name__ == '__main__':
    main()

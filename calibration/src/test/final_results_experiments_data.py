""" 
Final experiments on datasets 

Author: Thomas Mortier
Date: May 2022
"""
import sys
import warnings
warnings.filterwarnings("ignore")
import pickle
import time
import ast
import ternary
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pathlib import Path
sys.path.insert(0, "/home/data/tfmortier/Research/Calibration/calibration/src/main/py")
from caltest import tvdistance # distance functions
from caltest import aucb_obj, hl_obj, skce_ul_obj, skce_uq_obj, classece_obj, confece_obj # objectives 
from caltest import hl, skceul, skceuq, confece, classece # estimators
from caltest import npbetest
from pycalib.metrics import conf_ECE
from scipy.stats import halfnorm, dirichlet, multinomial, multivariate_normal
from scipy.optimize import linprog
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def main_experiments_data_plantclef():
    pref = "./pkl/"
    #data_list = ["cifar10", "cal101", "cal256", "plantclef", "bact"]
    data_list = ["plantclef"]
    arch_list = ["MOB", "VGG"]
    #mod_list = [("d1","0.1.pkl"),("d1","0.6.pkl"),("d10","0.1.pkl"),("d100","0.1.pkl"),("d10","0.6.pkl"),("d100","0.6.pkl"),("b",".pkl")]
    mod_list = [("d10","0.6.pkl"),("b",".pkl")]
    for d in data_list:
        for a in arch_list:
            for m in mod_list:
                # read in data
                try:
                    # read data
                    data_pkl = pickle.load(open(pref+d+m[0]+a+m[1],"rb"))
                    # split in calibration and test set
                    P = data_pkl["probs"]
                    y = data_pkl["labels"]
                    if len(P.shape) == 2:
                        P = np.expand_dims(data_pkl["probs"],axis=1)
                    # split data
                    P_test, P, y_test, y = train_test_split(P, y, test_size=0.5, random_state=2022)
                    # calculate performance of ensemble average
                    acc_avg = np.mean(np.argmax(np.mean(P,axis=1),axis=1)==y)
                    # evaluate confidence ECE
                    params = {
                        "optim": "cobyla",
                        "n_resamples": 100, 
                        "h": 2,
                        "dist": tvdistance,
                        "nbins": 5,
                        "alpha": 0.05,
                        "test": confece, 
                        "obj": confece_obj}
                    dec_confece, l_confece = npbetest(P_test, y_test, params)
                    confece_l = confece_obj(l_confece, P, y, params)
                    confece_avg = confece_obj(np.array([1/P.shape[1]]*P.shape[1]),P,y,params)
                    acc_confece = np.mean(np.argmax(np.matmul(np.swapaxes(P,1,2),l_confece),axis=1)==y)
                    # evaluate classwise ECE
                    params = {
                        "optim": "cobyla",
                        "n_resamples": 100, 
                        "h": 2,
                        "dist": tvdistance,
                        "nbins": 5,
                        "alpha": 0.05,
                        "test": classece, 
                        "obj": classece_obj}
                    dec_classece, l_classece = npbetest(P_test, y_test, params)
                    classece_l = classece_obj(l_classece, P, y, params)
                    classece_avg = classece_obj(np.array([1/P.shape[1]]*P.shape[1]),P,y,params)
                    acc_classece = np.mean(np.argmax(np.matmul(np.swapaxes(P,1,2),l_classece),axis=1)==y)
                    to_print = [d, a, m[0]+"_"+m[1], str(round(acc_avg,4)), str(round(confece_avg,4)), str(round(classece_avg,4)), str(dec_confece==1), str(round(acc_confece,4)), str(round(confece_l,4)), str(dec_classece==1), str(round(acc_classece,4)), str(round(classece_l,4))]
                    print(" ".join(to_print))
                except Exception as e:
                    print("Error for file {}".format(pref+d+m[0]+a+m[1]))
                    break

def main_experiments_data_prot():
    pref = "./pkl/"
    #data_list = ["cifar10", "cal101", "cal256", "plantclef", "bact"]
    data_list = ["prot"]
    arch_list = ["MOB"]
    #mod_list = [("d1","0.1.pkl"),("d1","0.6.pkl"),("d10","0.1.pkl"),("d100","0.1.pkl"),("d10","0.6.pkl"),("d100","0.6.pkl"),("b",".pkl")]
    #mod_list = [("d1","0.1.pkl"),("d10","0.1.pkl"),("d10","0.6.pkl"),("b",".pkl")]
    mod_list = [("b",".pkl")]
    for d in data_list:
        for a in arch_list:
            for m in mod_list:
                # read in data
                try:
                    # read data
                    data_pkl = pickle.load(open(pref+d+m[0]+a+m[1],"rb"))
                    # split in calibration and test set
                    P = data_pkl["probs"]
                    y = data_pkl["labels"]
                    if len(P.shape) == 2:
                        P = np.expand_dims(data_pkl["probs"],axis=1)
                    # split data
                    P_test, P, y_test, y = train_test_split(P, y, test_size=0.5, random_state=2022)
                    # calculate performance of ensemble average
                    acc_avg = np.mean(np.argmax(np.mean(P,axis=1),axis=1)==y)
                    # evaluate confidence ECE
                    params = {
                        "optim": "cobyla",
                        "n_resamples": 100, 
                        "h": 2,
                        "dist": tvdistance,
                        "nbins": 5,
                        "alpha": 0.05,
                        "test": confece, 
                        "obj": confece_obj}
                    dec_confece, l_confece = npbetest(P_test, y_test, params)
                    confece_l = confece_obj(l_confece, P, y, params)
                    confece_avg = confece_obj(np.array([1/P.shape[1]]*P.shape[1]),P,y,params)
                    acc_confece = np.mean(np.argmax(np.matmul(np.swapaxes(P,1,2),l_confece),axis=1)==y)
                    # evaluate classwise ECE
                    params = {
                        "optim": "cobyla",
                        "n_resamples": 100, 
                        "h": 2,
                        "dist": tvdistance,
                        "nbins": 5,
                        "alpha": 0.05,
                        "test": classece, 
                        "obj": classece_obj}
                    dec_classece, l_classece = npbetest(P_test, y_test, params)
                    classece_l = classece_obj(l_classece, P, y, params)
                    classece_avg = classece_obj(np.array([1/P.shape[1]]*P.shape[1]),P,y,params)
                    acc_classece = np.mean(np.argmax(np.matmul(np.swapaxes(P,1,2),l_classece),axis=1)==y)
                    to_print = [d, a, m[0]+"_"+m[1], str(round(acc_avg,4)), str(round(confece_avg,4)), str(round(classece_avg,4)), str(dec_confece==1), str(round(acc_confece,4)), str(round(confece_l,4)), str(dec_classece==1), str(round(acc_classece,4)), str(round(classece_l,4))]
                    print(" ".join(to_print))
                except Exception as e:
                    print("Error for file {}".format(pref+d+m[0]+a+m[1]))
                    break

def main_experiments_data():
    pref = "./pkl/"
    #data_list = ["cifar10", "cal101", "cal256", "plantclef", "bact"]
    data_list = ["plantclef", "prot"]
    arch_list = ["MOB", "VGG"]
    #mod_list = [("d1","0.1.pkl"),("d1","0.6.pkl"),("d10","0.1.pkl"),("d100","0.1.pkl"),("d10","0.6.pkl"),("d100","0.6.pkl"),("b",".pkl")]
    mod_list = [("d1","0.1.pkl"),("d10","0.1.pkl"),("d10","0.6.pkl"),("b",".pkl")]
    for d in data_list:
        for a in arch_list:
            for m in mod_list:
                # read in data
                try:
                    # read data
                    data_pkl = pickle.load(open(pref+d+m[0]+a+m[1],"rb"))
                    # split in calibration and test set
                    P = data_pkl["probs"]
                    y = data_pkl["labels"]
                    if len(P.shape) == 2:
                        P = np.expand_dims(data_pkl["probs"],axis=1)
                    # split data
                    P_test, P, y_test, y = train_test_split(P, y, test_size=0.5, random_state=2022)
                    # calculate performance of ensemble average
                    acc_avg = np.mean(np.argmax(np.mean(P,axis=1),axis=1)==y)
                    # evaluate confidence ECE
                    params = {
                        "optim": "cobyla",
                        "n_resamples": 100, 
                        "h": 2,
                        "dist": tvdistance,
                        "nbins": 5,
                        "alpha": 0.05,
                        "test": confece, 
                        "obj": confece_obj}
                    dec_confece, l_confece = npbetest(P_test, y_test, params)
                    confece_l = confece_obj(l_confece, P, y, params)
                    confece_avg = confece_obj(np.array([1/P.shape[1]]*P.shape[1]),P,y,params)
                    acc_confece = np.mean(np.argmax(np.matmul(np.swapaxes(P,1,2),l_confece),axis=1)==y)
                    # evaluate classwise ECE
                    params = {
                        "optim": "cobyla",
                        "n_resamples": 100, 
                        "h": 2,
                        "dist": tvdistance,
                        "nbins": 5,
                        "alpha": 0.05,
                        "test": classece, 
                        "obj": classece_obj}
                    dec_classece, l_classece = npbetest(P_test, y_test, params)
                    classece_l = classece_obj(l_classece, P, y, params)
                    classece_avg = classece_obj(np.array([1/P.shape[1]]*P.shape[1]),P,y,params)
                    acc_classece = np.mean(np.argmax(np.matmul(np.swapaxes(P,1,2),l_classece),axis=1)==y)
                    to_print = [d, a, m[0]+"_"+m[1], str(round(acc_avg,4)), str(round(confece_avg,4)), str(round(classece_avg,4)), str(dec_confece==1), str(round(acc_confece,4)), str(round(confece_l,4)), str(dec_classece==1), str(round(acc_classece,4)), str(round(classece_l,4))]
                    print(" ".join(to_print))
                except Exception as e:
                    print("Error for file {}".format(pref+d+m[0]+a+m[1]))
                    break
if __name__=="__main__":
    #print("PLANTCLEF")
    #main_experiments_data_plantclef()
    print("PROT")
    main_experiments_data_prot()

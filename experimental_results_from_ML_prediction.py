#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 12:02:26 2021

@author: joaquin
"""
import numpy as np
import pandas as pd
import matplotlib
import argparse
import torch.nn.functional as F
import torch as th
import ML_functions as MLF
import data_analysis_funcs as DAF
from sklearn.metrics import confusion_matrix
import seaborn as sns

# importing sys

import DenseNet_C
import ResNet_C


save_prob = False
#########################################
##############Pytorch model##############
#########################################
nnet ="densnet"
# noise = "no_noise"
noise = "LC"
# noise = "2_uncertainty"
# noise = "white_noise"
# noise = "4_uncertainty"
# noise = "both_noise"



dataset=3000
seed=2
epoch=1000
model = MLF.ML_model_read_func(nnet = nnet,
                       nnet_file = DenseNet_C,
                       dataset = dataset,
                       epoch = epoch,
                       seed = seed,
                       noise = noise, nnet_dir = '/home/osboxes/upm/race_tracking_project_4/data/neural_network_stored/')

########################################################
##############Loading Experimental dataSet##############
########################################################
rt0=[29,31,32,44,61,65]       
rt1=[39,40,42,43,47,51]         
rt2=[50]        
rt3=[24,26,27,45,46,69]         
rt4=[23,48,49,70,71,72] 

rt=rt1+rt2+rt3+rt4
p_exp_list = []
n_rt = len(rt)
n_classes = 4
RT_Class_list =[]
real_type_list = []
test_no_list = []
probability_matrix  = np.zeros((n_rt, n_classes))
for index,case in enumerate(rt):
    test_type, test_properties, sensprs = DAF.func_experimental_tests_parameters("../experimentalData/Tests Description.xlsx", case)
    exp_path ='exp_norm_WORT0_modified_11042022_235609'
    testData='/home/osboxes/Desktop/RunOpenFoam/87_RTM_Experimental_Analysis/modifiedExperimentalData/'+exp_path+'/'+str(test_type[-1])+'_test{}'.format(case)+'.xlsx'
    data_exp = pd.read_excel(testData)
    time = data_exp['time']
    p_exp = np.array(data_exp[['s0','s1','s2','s3','s4']])
############################################
###### Transformation experimental data #########
############################################

    x_test = p_exp.reshape(1, 1, p_exp.shape[0], p_exp.shape[1])
    x_test = MLF.square_image_transformation(imgs = x_test)
    ##############################################
    ##################Evaluation##################
    ##############################################
    device = "cpu"
    x_data_tensor = th.from_numpy(x_test).float().to(device)
    input = x_data_tensor.to(device)
    with th.no_grad():
        model.eval()
        output = model(input)
        y_prob = F.softmax(output, dim = 1)
        top_pred = y_prob.argmax(1, keepdim = True)
        y_prob_np = np.array(y_prob)
        probability_matrix[index,:] = y_prob_np


    if(top_pred == 0):
        RT_class=1
    elif(top_pred==1):
        RT_class=2
    elif(top_pred==2):
        RT_class=3
    elif(top_pred==3):
        RT_class=4
    # elif(top_pred==4):
    #     RT_class=4
    
    RT_Class_list.append(RT_class)
    RT_Class_np = np.array(RT_Class_list)
    real_type = int(test_type[-1][-1])
    real_type_list.append(real_type)
    real_type_np = np.array(real_type_list)
    test_no_list.append(test_type[0])
    test_no_np = np.array(test_no_list)
    
type_list =['type1','type2','type3', 'type4']
confusion_mat= confusion_matrix(real_type_list, RT_Class_list, labels=np.unique(real_type_list))
cm = pd.DataFrame(confusion_mat, np.array(type_list), np.array(type_list))
cm.to_csv('../confusion_Matrix/conf_{}_{}.csv'.format(noise,epoch))
sns.heatmap(cm, cmap= "YlGnBu", cbar=True, annot= True,
                linewidth=0.5)

# confusion_mat_DF.to_csv("conf_matrix.csv")

print(confusion_mat)
print(noise+'_'+ str(epoch))
correct_pred = []
wrong_pred = []
wrong_pred_type = []
wrong_real_type = []
for i in range(len(test_no_list)):
    if real_type_list[i] == RT_Class_list[i]:
        correct_pred.append(test_no_list[i])
    else:
        wrong_pred.append(test_no_list[i])
        wrong_pred_type.append(RT_Class_list[i])
        wrong_real_type.append(real_type_list[i])
result = np.array((wrong_pred,wrong_real_type,wrong_pred_type))
accuracy = len(correct_pred)/len(real_type_list)*100
print("accuracy = "+ str(accuracy))
output_dic = {'test_no':test_no_np, 'real': real_type_list, 'pred':RT_Class_np,"RT1_prob" : probability_matrix[:,0],"RT2_prob" : probability_matrix[:,1], "RT3_prob" : probability_matrix[:,2],"RT4_prob" : probability_matrix[:,3]}
output_DF = pd.DataFrame(output_dic)
if save_prob:
    output_DF.to_excel("./probability/prediction_"+str(nnet)+"_"+str(noise)+"_"+str(dataset)+"_v"+str(seed)+"_epoch_"+str(epoch)+".xlsx")


    
            
            


   

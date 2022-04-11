# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 17:28:53 2022

@author: keayvan.keramati
"""
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def func_experimental_tests_parameters(excel_file_path, case):
    TestsDescription_excel_file = pd.read_excel(excel_file_path)
    test_no = list(TestsDescription_excel_file['Test No'])
    index_test = test_no.index(case)
    test_date = TestsDescription_excel_file.loc[index_test]['Date']
    type_test=TestsDescription_excel_file.loc[index_test]['Type']
    K =1/TestsDescription_excel_file.loc[index_test]["K_fab"]
    mu =TestsDescription_excel_file.loc[index_test]['Mhu(mpa.s)']/1000
    L= 0.3
    end_time=int(TestsDescription_excel_file.loc[index_test]['end time'])
    sensors_coef_list = []
    s0_loc = TestsDescription_excel_file.loc[index_test]['S0']
    s1_loc = TestsDescription_excel_file.loc[index_test]['S1']
    s2_loc = TestsDescription_excel_file.loc[index_test]['S2']
    s3_loc = TestsDescription_excel_file.loc[index_test]['S3']
    s4_loc = TestsDescription_excel_file.loc[index_test]['S4']
    for i in range(5):
        sensors_coef = TestsDescription_excel_file.loc[index_test][24+i]
        sensors_coef_list.append(sensors_coef)
    return [case,index_test,test_date, type_test], [K, mu, L, end_time], [sensors_coef_list, s0_loc, s1_loc, s2_loc, s3_loc, s4_loc]

def loading_experimental_data(case, index_test,test_date, s0_loc, s1_loc, s2_loc, s3_loc, s4_loc):
    test_name="test"+str(case)
    file_name='Z:/ExpandSim/{}th Test {}/'.format(case,round(int(test_date)))+test_name+".ASC"
    file = pd.read_csv(file_name,sep="\t",header=0)
    columns=file.columns
    time=file[[columns[0]]]
    sensors=file[[columns[s0_loc],columns[s1_loc],columns[s2_loc],
                  columns[s3_loc],columns[s4_loc]]]
    data=np.array(sensors)
    time = np.array(time)
    time_f = time[:,0]    
    return time_f, data

def filteration_fun(raw_data):
    filtered_data=np.zeros((len(raw_data[:,0]),len(raw_data[0,:])))  
    for i in range(5):
        filtered_data[:,i] = savgol_filter(raw_data[:,i], 0.019*len(raw_data[:,i]), 1, mode = 'nearest')
    return filtered_data

def shift_fun(filtered_pressure_data):
        shifted_data=np.zeros((len(filtered_pressure_data[:,0]),len(filtered_pressure_data[0,:])))
        for i in range(5):
            shifted_data[:,i]=filtered_pressure_data[:,i]-np.min(filtered_pressure_data[:,i])
            shifted_data[shifted_data < 0]=0
        return shifted_data

def test_area_duration_data_func(corrolated_pressure,time, end_time):
    deravative=np.gradient(corrolated_pressure[:,0],time)
    deravative_filtered = savgol_filter(deravative, 0.02*len(deravative), 1, mode = 'nearest')
    deravative_n2=np.gradient(deravative_filtered ,time)
    max_start_n1 = np.argmax(deravative[:int(len(corrolated_pressure[:,0])/2)])
    max_start_n2 = np.argmax(deravative_n2[:int(len(corrolated_pressure[:,0])/2)])
    max_start = min(max_start_n1, max_start_n2)   
    modified_pressure=corrolated_pressure[max_start+1:max_start+1+end_time*10,:]
    time_new1=np.array(time[max_start+1:max_start+1+end_time*10])
    time_new = time_new1-np.min(time_new1)
    return  time_new, modified_pressure 

def normalizeion_fun(modified_pressure, time_array, mu, L, K, p_max):
    tff = mu*L*L/(2*K*p_max*1e5) 
    time_normalized = time_array/tff
    normalized_pressure = modified_pressure / np.max(modified_pressure[:,0])
    return time_normalized, normalized_pressure, tff
    
def correlation_func(shifted_data,sensors_coef_list):
     corrolated_pressure=np.zeros((len(shifted_data[:,0]),len(shifted_data[0,:])))
     for i in range(5):
        coef_sensor = sensors_coef_list[i]
        corrolated_pressure[:,i] = [element * coef_sensor for element in shifted_data[:,i]]
     p_max = np.max(corrolated_pressure[:,0])
     return corrolated_pressure, p_max

def extrapolation_fun(x, y, min_time, max_time, n_time_step):
     p_extrapolate = np.zeros((n_time_step,len(y[0,:])))
     time_norm = np.linspace(min_time, max_time,n_time_step) 
     for i in range(len(y[0,:])):
         
         interpolat_func = interp1d(x, y[:,i], kind='nearest', bounds_error = False, fill_value="extrapolate")
         time_norm = np.linspace(min_time, max_time,n_time_step) 
         p_norm = interpolat_func(time_norm)
         p_extrapolate[:,i] = p_norm
     return time_norm, p_extrapolate 
 
def filling_time_exponentioal_func(t ,gamma):
    t_filling_real = t+((1/gamma)*(np.exp(-gamma*t)-1))
    return t_filling_real
def exponential_polynomial_func(x, a, b, c, d, e, f):
        return  a * np.exp(1-b*(d*x**2+f*x))+ c

def exponentioal_1D_fitting_func(time, p, time_new):
    def func1D(x, a, gamma):
        return  a * (1-np.exp(-gamma * x))
    popt,_ = curve_fit(func1D, time, p, maxfev = 1000)
    p_max, gamma = popt
    z_1D = func1D(time_new,p_max,gamma)
    return time_new, z_1D, [p_max, gamma]

def permeability_func(t_exp ,gamma, mu, L, p_max):
    t_filling_real = t_exp+(1/gamma)*(1-(np.exp(-gamma*t_exp)))
    k_fabric = mu*L*L/(2*t_filling_real*p_max*1e5)
    return k_fabric
def coef_fun(pressure_shifted, normalized_pressure):
    max_p_list=[]
    max_final_data=[]
    for i in range(len(pressure_shifted[:,0])):
        max_p = np.max(pressure_shifted[:,i])
        max_p_list.append(max_p)
        max_final_data = np.max(normalized_pressure[:,i])
        max_final_data.append(max_final_data)  
    coef_list=[]
    for i in range(len(max_p_list)):
        coef = max_p_list[0]/max_p_list[i]
        coef_list.append(coef)
    return coef_list

def data_after_raising(y, x):
    deravative = np.gradient(y,0.1)
    max_start = np.argmax(deravative)
    y_new = y[max_start-1:]
    x_new = x[max_start-1:]

    return  y_new, x_new, max_start
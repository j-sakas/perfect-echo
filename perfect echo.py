# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:57:34 2021

@author: justi
"""
import numpy as np
import matplotlib.pyplot as plt

class Spin:
    def __init__(self,name,JHH,JHF):
        self.name = name
        while len(JHH) < 5:
            JHH.append(0)
        self.JHH = JHH
        self.JHF = JHF


#parameters 
time = 100/1000 #100 ms - x axis
delta = 20/1000 #delta delay - 20 ms

def transfer_func(spin,time = time): #plots transfer fuction for perfect echo with HF evolution
    
    x = np.arange(0, time, 0.0001)
    pi = np.pi
    JHF = spin.JHF
    JHH = spin.JHH
    if JHF == 0:
        HF_factor = 0#1
    else:
        HF_factor = np.sin(np.pi*JHF*x)
    term1 = np.cos(np.pi*x*JHH[0])**2 * np.cos(np.pi*JHH[1]*x)**2 * np.cos(np.pi*JHH[2]*x)**2 * np.cos(np.pi*JHH[3]*x)**2
    term2 = np.sin(pi*JHH[0]*x)**2 * np.cos(pi*JHH[1]*x) * np.cos(pi*JHH[2]*x) * np.cos(pi*JHH[3]*x) * np.cos(pi*JHH[4]*x) 
    term3 = np.sin(pi*JHH[1]*x)**2 * np.cos(pi*JHH[0]*x) * np.cos(pi*JHH[2]*x) * np.cos(pi*JHH[3]*x) * np.cos(pi*JHH[4]*x) 
    term4 = np.sin(pi*JHH[2]*x)**2 * np.cos(pi*JHH[1]*x) * np.cos(pi*JHH[0]*x) * np.cos(pi*JHH[3]*x) * np.cos(pi*JHH[4]*x) 
    term5 = np.sin(pi*JHH[3]*x)**2 * np.cos(pi*JHH[1]*x) * np.cos(pi*JHH[2]*x) * np.cos(pi*JHH[0]*x) * np.cos(pi*JHH[4]*x) 
    term6 = np.sin(pi*JHH[4]*x)**2 * np.cos(pi*JHH[1]*x) * np.cos(pi*JHH[2]*x) * np.cos(pi*JHH[3]*x) * np.cos(pi*JHH[0]*x) 
    
    y = (term1 + term2 + term3 + term4 + term5 + term6) * HF_factor
    
    
    return  plt.plot(x,y,label=spin.name)


def transfer_func_delta(spin,delta = delta, time=time): #plots transfer function with delta delay for second spin echo
    
    x = np.arange(0, time, 0.0001)
    pi = np.pi
    JHF = spin.JHF
    JHH = spin.JHH
    delta_factor = np.sin(pi*JHF*(x+delta)) * np.cos(pi*JHH[0]*delta)* np.cos(pi*JHH[1]*delta)* np.cos(pi*JHH[2]*delta) * np.cos(pi*JHH[3]*delta)
    
    term1 = np.cos(np.pi*x*JHH[0])**2 * np.cos(np.pi*JHH[1]*x)**2 * np.cos(np.pi*JHH[2]*x)**2 * np.cos(np.pi*JHH[3]*x)**2
    term2 = np.sin(pi*JHH[0]*x)**2 * np.cos(pi*JHH[1]*x) * np.cos(pi*JHH[2]*x) * np.cos(pi*JHH[3]*x) * np.cos(pi*JHH[4]*x) 
    term3 = np.sin(pi*JHH[1]*x)**2 * np.cos(pi*JHH[0]*x) * np.cos(pi*JHH[2]*x) * np.cos(pi*JHH[3]*x) * np.cos(pi*JHH[4]*x) 
    term4 = np.sin(pi*JHH[2]*x)**2 * np.cos(pi*JHH[1]*x) * np.cos(pi*JHH[0]*x) * np.cos(pi*JHH[3]*x) * np.cos(pi*JHH[4]*x) 
    term5 = np.sin(pi*JHH[3]*x)**2 * np.cos(pi*JHH[1]*x) * np.cos(pi*JHH[2]*x) * np.cos(pi*JHH[0]*x) * np.cos(pi*JHH[4]*x) 
    term6 = np.sin(pi*JHH[4]*x)**2 * np.cos(pi*JHH[1]*x) * np.cos(pi*JHH[2]*x) * np.cos(pi*JHH[3]*x) * np.cos(pi*JHH[0]*x) 
    
    y = (term1 + term2 + term3 + term4 + term5 + term6) * delta_factor
    
    return plt.plot(x,y,label=spin.name)
    

def get_xy_data(plot):
    x = (plot[0][0].get_xdata())
    xy = np.empty([len(x),len(plot)+1])
    xy[:,0] = x
    for i in list(range(len(plot))):
        xy[:,i+1] = plot[i][0].get_ydata()
         
    return xy
    

#alpha glucose
s1 = Spin('H1',[3.9],3.7) #arguments are spin name, list of JHH couplings, JHF coupling
s2 = Spin('H2',[3.9,9.4],13.1)
s3 = Spin('H3',[9.5,8.9],54.3)
s4 = Spin('H4',[8.9,10.2],13.8)
s5 = Spin('H5',[10.2,2.3,4.9],0)
s6 = Spin('H6',[2.3,12.2],1.8)
s6a = Spin("H6'",[5.,12.3],0)

#betaglucose
s1b = Spin('H1',[8.0],0)
s2b = Spin('H2',[8.0,9.1],13.7)
s3b = Spin('H3',[9.1,8.8],52.9)
s4b = Spin('H4',[8.9,10.],13.8)
s5b = Spin('H5',[10.0,2.2,5.6],1.3)
s6b = Spin('H6',[2.2,12.4],1.5)
s6ab = Spin("H6'",[5.5,12.4],0)


#plot1
plt.figure()
plot1 = [transfer_func(s1),
transfer_func(s2),
transfer_func(s3),
transfer_func(s4),
transfer_func(s5),
transfer_func(s6),
transfer_func(s6a)]
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
plt.title('PE transfer function alpha-glucose')
xy_alpha = get_xy_data(plot1)
#plot2
plt.figure()
plot2= [transfer_func(s1b),
transfer_func(s2b),
transfer_func(s3b),
transfer_func(s4b),
transfer_func(s5b),
transfer_func(s6b),
transfer_func(s6ab)]
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
plt.title('PE transfer function beta-glucose')
xy_beta = get_xy_data(plot2)
#plot 3
plt.figure()
plot3 = [transfer_func_delta(s1),
transfer_func_delta(s2),
transfer_func_delta(s3),
transfer_func_delta(s4),
transfer_func_delta(s5),
transfer_func_delta(s6),
transfer_func_delta(s6a)]
plt.legend()
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
plt.title('PE transfer function alpha-glucose with delta=' + str(delta))
xy_alpha_with_delta = get_xy_data(plot3)
#plot 4
plt.figure()
plot4 = [transfer_func_delta(s1b),
transfer_func_delta(s2b),
transfer_func_delta(s3b),
transfer_func_delta(s4b),
transfer_func_delta(s5b),
transfer_func_delta(s6b),
transfer_func_delta(s6ab)]
plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
plt.title('PE transfer function beta-glucose with delta=' + str(delta))
xy_beta_with_delta = get_xy_data(plot4)



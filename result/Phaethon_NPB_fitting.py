# Derive Polarimetric Paramter



#==========================================
#Import packages and define the function
#==========================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import theano
import pymc3 as pm


def d2r(x): return np.deg2rad(x)
def r2d(x): return np.rad2deg(x)
def sin(x): return np.sin(x)
def cos(x): return np.cos(x)
def tan(x): return np.tan(x)
def log(x): return np.log(x)
def log10(x): return np.log10(x)
def exp(x):
    return np.exp(x)
def ln(x):
    return np.log(x)
def trigonal_function(alpha,h,a0,c1,c2):
    alpha_rad = d2r(alpha)
    a0_rad = d2r(a0)
    D2R = 3.14/180
    P_alpha = (h / D2R 
             * (sin(alpha_rad) / sin(a0_rad))**c1 
             * (cos(alpha_rad / 2) / cos(a0_rad/ 2))**c2 
             * sin(alpha_rad - a0_rad)
            )
    return P_alpha
def exponential(x,a,b,c):
    eq = a*np.exp(-x/b) -a + c*x
    return eq 



#==========================================
#Bring the data
#==========================================

df = pd.read_csv('Phaethon_data.csv')




#==========================================
#Set the initial  condition
#==========================================
#Phase angle range
df_tri = df[df['alpha']<30] #For Trigonal
df_exp = df[df['alpha']<30] #For Exponential


#Trigonal
Boundary_tri = pd.DataFrame({'h':[0,1], #lower , upper boundary
                           'a0':[10,30],
                           'c1':[0,10],
                           'c2':[0,10]})
p0_tri = [0.10, 20,0.67, 0.1]

#EXPONENTIAL
Boundary_exp = pd.DataFrame({'a':[10,20], #lower , upper boundary
                           'b':[15,25],
                           'c':[0,1]})
p0_exp = [ 15.22876846, 18.71077083,  0.49865412]











#==========================================
# Start fitting
#==========================================


# Trigonal ================================
D2R = 3.14/180
basic_model_tri = pm.Model()

alpha = df_tri['alpha'].values
P = df_tri['P'].values
eP = df_tri['eP'].values

with basic_model_tri:
    h = pm.Uniform('h',  Boundary_tri['h'].values[0],  Boundary_tri['h'].values[1])
    a0 = pm.Uniform('a0', Boundary_tri['a0'].values[0], Boundary_tri['a0'].values[1])
    c1 = pm.Uniform('c1', Boundary_tri['c1'].values[0], Boundary_tri['c1'].values[1])
    c2 = pm.Uniform('c2', Boundary_tri['c2'].values[0], Boundary_tri['c2'].values[1])

    sigma = theano.shared(np.asarray(eP, dtype=theano.config.floatX), name='sigma')
    P_alpha = trigonal_function(alpha,h,a0,c1,c2)

    liklihood = pm.Normal('likelihood', mu=P_alpha, sd=sigma, observed=P)
    trace_tri = pm.sample(20000, start=dict(h=p0_tri[0],
                                           a0=p0_tri[1],
                                           c1=p0_tri[2], 
                                           c2=p0_tri[3]))
    
summary_tri = pm.summary(trace_tri, hdi_prob=0.6827).round(5)   
pm.traceplot(trace_tri)
pm.summary(trace_tri, hdi_prob=0.6827).round(5)     


#Deriving Pmin & Slope h
from scipy.stats import chi2
samples_tri = np.array([trace_tri.get_values(key) for key in ['h','a0','c1','c2']]).T

delta = chi2.ppf(0.6827, len(['h','a0','c1','c2']))
n_trace_tri = trace_tri['h'].shape[0]
log_tri =  dict(h=trace_tri['h'], a0=trace_tri['a0'], c1=trace_tri['c1'], c2=trace_tri['c2'])

Pmin_Tri_mcmc_dist = []
amin_Tri_mcmc_dist = []
hmin_Tri_mcmc_dist = []

for i in range(len(trace_tri['h'])):
    h_i = log_tri['h'][i]
    a0_i = log_tri['a0'][i]
    c1_i = log_tri['c1'][i]
    c2_i = log_tri['c2'][i]
    
    xxx = np.arange(5,15,0.01)
    P_i = trigonal_function(xxx,h_i,a0_i,c1_i,c2_i)
    Pmin_i = min(P_i)
    order = list(P_i).index(Pmin_i)
    amin_i = xxx[order]
    Pmin_Tri_mcmc_dist.append(Pmin_i)
    amin_Tri_mcmc_dist.append(amin_i)
    hmin_Tri_mcmc_dist.append(h_i)

Pmin_Tri_mcmc = np.median(Pmin_Tri_mcmc_dist)
ePmin_Tri_mcmc_upper = np.percentile(Pmin_Tri_mcmc_dist,84.135) - Pmin_Tri_mcmc
ePmin_Tri_mcmc_lower = Pmin_Tri_mcmc - np.percentile(Pmin_Tri_mcmc_dist,15.865)

amin_Tri_mcmc = np.median(amin_Tri_mcmc_dist)
eamin_Tri_mcmc_upper = np.percentile(amin_Tri_mcmc_dist,84.135) - amin_Tri_mcmc
eamin_Tri_mcmc_lower = amin_Tri_mcmc - np.percentile(amin_Tri_mcmc_dist,15.865)

hmin_Tri_mcmc = np.median(hmin_Tri_mcmc_dist)
ehmin_Tri_mcmc_upper = np.percentile(hmin_Tri_mcmc_dist,84.135) - hmin_Tri_mcmc
ehmin_Tri_mcmc_lower = hmin_Tri_mcmc - np.percentile(hmin_Tri_mcmc_dist,15.865)

 
# Linear-exponential  ================================
alpha = df_exp['alpha'].values
P = df_exp['P'].values
eP = df_exp['eP'].values

basic_model_exp = pm.Model()
with basic_model_exp:
    a = pm.Uniform('a',  Boundary_exp['a'].values[0],  Boundary_exp['a'].values[1])
    b = pm.Uniform('b', Boundary_exp['b'].values[0], Boundary_exp['b'].values[1])
    c = pm.Uniform('c', Boundary_exp['c'].values[0], Boundary_exp['c'].values[1])
    sigma = theano.shared(np.asarray(eP, dtype=theano.config.floatX), name='sigma')
    P_alpha = exponential(alpha,a,b,c)
    

    liklihood = pm.Normal('likelihood', mu=P_alpha, sd=sigma, observed=P)
    start = {}
    trace_exp = pm.sample(20000, start=dict(a = p0_exp[0],
                                           b = p0_exp[1],
                                           c = p0_exp[2]))
    
summary_exp = pm.summary(trace_exp, hdi_prob=0.6827).round(5)   
pm.traceplot(trace_exp)
print('Exponential')
pm.summary(trace_exp, hdi_prob=0.6827).round(5)     



#Deriving Pmin & Slope h
samples_exp = np.array([trace_exp.get_values(key) for key in ['a','b','c']]).T
n_trace_exp = trace_exp['a'].shape[0]

log_exp =  dict(a=trace_exp['a'], b=trace_exp['b'], c=trace_exp['c'])

Pmin_Exp_mcmc_dist = []
amin_Exp_mcmc_dist = []
a0_Exp_mcmc_dist = []
h_Exp_mcmc_dist = []

for i in range(len(trace_exp['a'])):
    a_i = log_exp['a'][i]
    b_i = log_exp['b'][i]
    c_i = log_exp['c'][i]
    
    xxx = np.arange(0,17,0.01)
    P_i = exponential(xxx,a_i,b_i,c_i)
    Pmin_i = min(P_i)
    order = list(P_i).index(Pmin_i)
    amin_i = xxx[order]
    Pmin_Exp_mcmc_dist.append(Pmin_i)
    amin_Exp_mcmc_dist.append(amin_i)
    
    
    xxx = np.arange(15,25,0.01)
    P_i = abs(exponential(xxx,a_i,b_i,c_i))
    order = list(P_i).index(min(P_i))
    a0_i = xxx[order]
    slope_i = (exponential(a0_i+0.01,a_i,b_i,c_i)-exponential(a0_i-0.01,a_i,b_i,c_i))/0.02
    
    a0_Exp_mcmc_dist.append(a0_i)
    h_Exp_mcmc_dist.append(slope_i)
    
Pmin_Exp_mcmc = np.median(Pmin_Exp_mcmc_dist)
ePmin_Exp_mcmc_upper = np.percentile(Pmin_Exp_mcmc_dist,84.135) - Pmin_Exp_mcmc
ePmin_Exp_mcmc_lower = Pmin_Exp_mcmc - np.percentile(Pmin_Exp_mcmc_dist,15.865)

amin_Exp_mcmc = np.median(amin_Exp_mcmc_dist)
eamin_Exp_mcmc_upper = np.percentile(amin_Exp_mcmc_dist,84.135) - amin_Exp_mcmc
eamin_Exp_mcmc_lower = amin_Exp_mcmc - np.percentile(amin_Exp_mcmc_dist,15.865)

h_Exp_mcmc = np.median(h_Exp_mcmc_dist)
eh_Exp_mcmc_upper = np.percentile(h_Exp_mcmc_dist,84.135) - h_Exp_mcmc
eh_Exp_mcmc_lower = h_Exp_mcmc - np.percentile(h_Exp_mcmc_dist,15.865)


a0_Exp_mcmc = np.median(a0_Exp_mcmc_dist)
ea0_Exp_mcmc_upper = np.percentile(a0_Exp_mcmc_dist,84.135) - a0_Exp_mcmc
ea0_Exp_mcmc_lower = a0_Exp_mcmc - np.percentile(a0_Exp_mcmc_dist,15.865)















#==========================================================
# Summarize the derive Pmin, Slope h, inversion, alpha_min
#==========================================================


MCMC_Result = pd.DataFrame({}) #Parameter, slope h, Pmin, alpha_min, inversion_angle
Function = ['Trigonal','Linear-exp']

Param_Tri = summary_tri['mean'].values
Param_exp = summary_exp['mean'].values



param_i = [Param_Tri, Param_exp]
Pmin_i = [Pmin_Tri_mcmc,Pmin_Exp_mcmc]
Pmin_upper_i = [ePmin_Tri_mcmc_upper,ePmin_Exp_mcmc_upper]
Pmin_lower_i = [ePmin_Tri_mcmc_lower,ePmin_Exp_mcmc_lower]

amin_i = [amin_Tri_mcmc,amin_Exp_mcmc]
eamin_upper_i = [eamin_Tri_mcmc_upper,eamin_Exp_mcmc_upper]
eamin_lower_i = [eamin_Tri_mcmc_lower,eamin_Exp_mcmc_lower]

slope_i = [summary_tri['mean']['h'], h_Exp_mcmc]
eslope_upper_i =  [summary_tri['hdi_84.135%']['h']-summary_tri['mean']['h'],
                    eh_Exp_mcmc_upper]
eslope_lower_i =  [summary_tri['mean']['h']-summary_tri['hdi_15.865%']['h'],
                    eh_Exp_mcmc_lower]

a0_i = [summary_tri['mean']['a0'], a0_Exp_mcmc]
ea0_upper_i =  [summary_tri['hdi_84.135%']['a0']-summary_tri['mean']['a0'],
                        ea0_Exp_mcmc_upper]
ea0_lower_i =  [summary_tri['mean']['a0']-summary_tri['hdi_15.865%']['a0'],
                        ea0_Exp_mcmc_lower]
    

for n,func in enumerate(['Trigonal','Linear-exp']) :

    MCMC_Result = MCMC_Result.append({'Method':'MCMC',
                                     'Function':func,
                                     'Parameter':param_i[n],
                                     'Pmin':Pmin_i[n],
                                     '+sigma Pmin':Pmin_upper_i[n],
                                     '-sigma Pmin':Pmin_lower_i[n],
                                     'alpha_min':amin_i[n],
                                     '+sigma alpha_min':eamin_upper_i[n],
                                     '-sigma alpha_min':eamin_lower_i[n],
                                     'h':slope_i[n],
                                     '+sigma h':eslope_upper_i[n],
                                     '-sigma h':eslope_lower_i[n],
                                     'a0':a0_i[n],
                                     '+sigma a0':ea0_upper_i[n],
                                     '-sigma a0':ea0_lower_i[n]},
                                    ignore_index=True)
new_index = ['Method', 'Function', 'Parameter', 
             'Pmin','+sigma Pmin','-sigma Pmin',
            'alpha_min','+sigma alpha_min','-sigma alpha_min',
            'h','+sigma h','-sigma h',
            'a0','+sigma a0','-sigma a0']


MCMC_Result=MCMC_Result.reindex(columns=new_index)
P_round = 2
alpha_round = 1
h_round = 3
MCMC_Result = MCMC_Result.round({'Pmin':P_round,'+sigma Pmin':P_round,'-sigma Pmin':P_round,
                                  'alpha_min':alpha_round,'+sigma alpha_min':alpha_round,'-sigma alpha_min':alpha_round,
                                  'h':h_round,'+sigma h':h_round,'-sigma h':h_round,
                                  'a0':alpha_round,'+sigma a0':alpha_round,'-sigma a0':alpha_round})



'''
RESULT 

  Method    Function                      Parameter  Pmin  +sigma Pmin  -sigma Pmin  alpha_min  +sigma alpha_min  -sigma alpha_min\      \
0   MCMC    Trigonal  [0.224, 19.885, 0.854, 5.455] -1.33         0.07         0.08        9.0               0.7               0.8  
1   MCMC  Linear-exp        [16.809, 19.811, 0.531] -1.32         0.06         0.06        9.2               0.1               0.2 


       h    +sigma h  -sigma h    a0  +sigma a0  -sigma a0  
0  0.224       0.011     0.017  19.9        0.3        0.3  
1  0.223       0.009     0.009  19.9        0.2        0.2  


'''


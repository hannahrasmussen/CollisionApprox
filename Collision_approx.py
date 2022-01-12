#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from nu_e_coll import nu_e_collisions as ve
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit


# In[2]:


x_values, w_values = np.polynomial.laguerre.laggauss(100) 
Gf = 1.166*10**-11 
me = .511 
g = 2 


@nb.jit(nopython=True)
def f(p,Tcm,c): 
    return 1/(np.e**(c*p/Tcm)+1)

@nb.jit(nopython=True)
def f_eq(p,T,eta): 
    return 1/(np.e**((p/T)-eta)+1)

@nb.jit(nopython=True)
def n_e(T): 
    E_array = np.sqrt(x_values**2 + me**2) 
    integral = np.sum((np.e**x_values)*w_values*(x_values**2)/(np.e**(E_array/T)+1))
    return (g/(2*np.pi**2))*integral 


# In[3]:


def cs_eta(T,Tcm,c):
    eta_arr = np.linspace(-10,10,1000)
    num_den_arr = np.zeros(len(eta_arr)) 
    hold = np.zeros(len(x_values))
    for i in range (len(eta_arr)):
        for j in range (len(x_values)):
            hold[j] = (np.e**x_values[j])*w_values[j]*(x_values[j]**2)*f_eq(x_values[j],T,eta_arr[i])
        num_den_arr[i] = (1/(2*np.pi**3)) * np.sum(hold) 
    cs = CubicSpline(num_den_arr,eta_arr) 
    
    integrand = np.zeros(len(x_values))
    for i in range (len(x_values)):
        integrand[i] = (np.e**x_values[i])*w_values[i]*(x_values[i]**2)*f(x_values[i],Tcm,c)
    num_den = (1/(2*np.pi**3)) * np.sum(integrand)
    eta = cs(num_den)
    return eta


# In[4]:


def model_An(a,T,c,npts=201,etop=20): 
    
    e_arr = np.linspace(0,etop,int(npts))
    bx = e_arr[1]-e_arr[0]
    eta = cs_eta(T,1/a,c)
    ne = n_e(T)
    p_arr = e_arr / a
    f_arr = f(p_arr,1/a,c)
    feq_arr = f_eq(p_arr,T,eta)
    
    def C_local(p_arr,A,n):
        C_arr = np.zeros(len(p_arr))
        for i in range (len(C_arr)):
            C_arr[i] = (p_arr[i]**n)*(f_arr[i]-feq_arr[i])
        C_arr = -A*ne*(Gf**2)*(T**(2-n))*C_arr
        return C_arr
    
    net = ve.driver(p_arr,T,f_arr,bx*(1/a))
    popt, pcov = curve_fit(C_local,p_arr[:int(0.5*len(p_arr))],net[:int(0.5*len(p_arr))])
    A,n = popt

    return A,n

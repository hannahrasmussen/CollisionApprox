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
    integral_arr = np.zeros(len(eta_arr)) #cubic spline integral that match w/ etas in the eta_array
    hold = np.zeros(len(x_values))
    for i in range (len(eta_arr)):
        for j in range (len(x_values)):
            hold[j] = (np.e**x_values[j])*w_values[j]*(x_values[j]**2)*f_eq(x_values[j],T,eta_arr[i])
        integral_arr[i] = np.sum(hold) 
    cs = CubicSpline(integral_arr,eta_arr) #cs will be different each time, depends on T
    
    integrand = np.zeros(len(x_values))
    for i in range (len(x_values)):
        integrand[i] = (np.e**x_values[i])*w_values[i]*(x_values[i]**2)*f(x_values[i],Tcm,c)
    integral = np.sum(integrand) #value that will match with the eta we eventually output from this function
    eta = cs(integral)
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
    net = ve.driver(p_arr,T,f_arr,bx*(1/a))
    
    def C_local(p_arr,A,n):
        C_arr = np.zeros(len(p_arr))
        for i in range (len(C_arr)):
            C_arr[i] = (p_arr[i]**n)*(f_arr[i]-feq_arr[i])
        C_arr = -A*ne*(Gf**2)*(T**(2-n))*C_arr
        return C_arr
   
    popt, pcov = curve_fit(C_local,p_arr[:int(0.5*len(p_arr))],net[:int(0.5*len(p_arr))])
    A,n = popt

    return A,n


#In[5]:


def cs_eta_new(T,Tcm,c,p_arr,f_arr):
    eta_arr = np.linspace(-10,10,1000)
    integral_arr = np.zeros(len(eta_arr)) #cubic spline integral that match w/ etas in the eta_array
    hold = np.zeros(len(x_values))
    for i in range (len(eta_arr)):
        for j in range (len(x_values)):
            hold[j] = (np.e**x_values[j])*w_values[j]*(x_values[j]**2)*f_eq(x_values[j],T,eta_arr[i])
        integral_arr[i] = np.sum(hold) 
    cs = CubicSpline(integral_arr,eta_arr) #cs will be different each time, depends on T
    
    integrand = np.zeros(len(p_arr))
    for i in range (len(p_arr)):
        integrand[i] = (p_arr[i]**2)*f(p_arr[i],Tcm,c)
    integral = np.sum(integrand) #value that will match with the eta we eventually output from this function
    eta = cs(integral)
    return eta

def model_An_new(a,T,c,e_arr,f_arr,npts=201,etop=20): 
    
    bx = e_arr[1]-e_arr[0]
    ne = n_e(T)
    p_arr = e_arr / a
    eta = cs_eta_new(T,1/a,c,p_arr,f_arr)
    feq_arr = f_eq(p_arr,T,eta)
    net = ve.driver(p_arr,T,f_arr,bx*(1/a))
    
    def C_local(p_arr,A,n):
        C_arr = np.zeros(len(p_arr))
        for i in range (len(C_arr)):
            C_arr[i] = (p_arr[i]**n)*(f_arr[i]-feq_arr[i])
        C_arr = -A*ne*(Gf**2)*(T**(2-n))*C_arr
        return C_arr
   
    popt, pcov = curve_fit(C_local,p_arr[:int(0.5*len(p_arr))],net[:int(0.5*len(p_arr))])
    A,n = popt

    return A,n

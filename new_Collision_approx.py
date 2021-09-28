#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
from nu_e_coll import new_ve_Collisions1_interp_extrap_0210 as ve
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
import numba as nb


# In[2]:


x_values, w_values = np.polynomial.laguerre.laggauss(100) 

#constants:
Gf = 1.166*10**-11 #This is the fermi constant in units of MeV^-2
me = .511          #Mass of an electron in MeV
g = 2           #mulitplicity, spin up and spin down


@nb.jit(nopython=True)
def f(p,Tcm,c): #occupation fraction for the neutrinos we've been using
    return 1/(np.e**(c*p/Tcm)+1)
@nb.jit(nopython=True)
def f_eq(p,T,eta): #occupation fraction of the neutrinos if they were in thermal equilibrium with the electrons & positrons
    return 1/(np.e**((p/T)-eta)+1)
@nb.jit(nopython=True)
def n_e(T): #number density of electrons
    Ep_array = np.sqrt(x_values**2 + me**2) #x_values are momenta here
    integral = np.sum((np.e**x_values)*w_values*(x_values**2)/(np.e**(Ep_array/T)+1))
    return (g/(2*np.pi**2))*integral 


# In[3]:


def cs_eta(T,Tcm,c):
    integrand = np.zeros(len(x_values))
    for i in range (len(x_values)):
        integrand[i] = (np.e**x_values[i])*w_values[i]*(x_values[i]**2)*f(x_values[i],Tcm,c)
    integral = np.sum(integrand) #value that will match with the eta we eventually output from this function
    
    Eta_array = np.linspace(-10,10,1000) #trying to make a range of etas that will encompass any potential eta
    integral_array = np.zeros(len(Eta_array)) #cubic spline integral that match w/ etas in the eta_array
    hold = np.zeros(len(x_values))
    for i in range (len(Eta_array)):
        for j in range (len(x_values)):
            hold[j] = (np.e**x_values[j])*w_values[j]*(x_values[j]**2)*f_eq(x_values[j],T,Eta_array[i])
        integral_array[i] = np.sum(hold) 
    
    cs = CubicSpline(integral_array,Eta_array) #cs actually will be different each time, depends on T
    eta = cs(integral)
    return eta



# In[4]:


def An(p_array,a,T,c): 
    
    temps = np.array([T,1/a])
    
    e_array = p_array*a
    boxsize = e_array[1]-e_array[0]
    eta = cs_eta(T,1/a,c)
    
    ne = n_e(T)
    f_array = f(p_array,1/a,c)
    feq_array = f_eq(p_array,T,eta)
    net = ve.driver(p_array,T,f_array,boxsize*(1/a))
    #frs = ve2.driver(p_array,T,f_array,boxsize*(1/a))

    def C_local(p_array,A,n):
        C_array = np.zeros(len(p_array))
        for i in range (len(C_array)):
            C_array[i] = (p_array[i]**n)*(f_array[i]-feq_array[i])
        C_array = -A*ne*(Gf**2)*(T**(2-n))*C_array
        return C_array
   
    popt, popc = curve_fit(C_local,e_array[:int(0.5*len(e_array))]*(1/a),net[:int(0.5*len(e_array))])
    C_array = C_local(e_array*(1/a),*popt)
    A,n = popt

    #plt.figure()
    #plt.plot(e_array, net, 'gold', label='Net Rate')
    #plt.plot(e_array, C_array, 'orange', label='Approx. Net Rate: A=%5.3f, n=%5.3f' % tuple(popt) )
    #plt.plot(e_array, frs, 'red', label='FRS Rate')
    #plt.plot(e_array, net/frs, 'magenta', label='Ratio of Net & FRS Rates')
    #plt.xlabel("Epsilon values")
    #plt.ylabel("Rate")
    #plt.title('T=%5.3f and T_cm=%5.3f' % tuple(temps))
    #plt.legend(loc='lower right')
    #plt.show()
    
    return A,n,C_array

def model_An(a,T,c,npts=201,etop=20): 
    
    e_array = np.linspace(0,etop,int(npts))
    boxsize = e_array[1]-e_array[0]
    eta = cs_eta(T,1/a,c)
    
    ne = n_e(T)
    
    p_array = e_array / a
    f_array = f(p_array,1/a,c)
    feq_array = f_eq(p_array,T,eta)
    net = ve.driver(p_array,T,f_array,boxsize*(1/a))
#    frs = ve2.driver(p_array,T,f_array,boxsize*(1/a))

    def C_local(p_array,A,n):
        C_array = np.zeros(len(p_array))
        for i in range (len(C_array)):
            C_array[i] = (p_array[i]**n)*(f_array[i]-feq_array[i])
        C_array = -A*ne*(Gf**2)*(T**(2-n))*C_array
        return C_array
   
    popt, popc = curve_fit(C_local,e_array[:int(0.5*len(e_array))]*(1/a),net[:int(0.5*len(e_array))])
#    C_array = C_local(e_array*(1/a),*popt)
    A,n = popt

    #plt.figure()
    #plt.plot(e_array, net, 'gold', label='Net Rate')
    #plt.plot(e_array, C_array, 'orange', label='Approx. Net Rate: A=%5.3f, n=%5.3f' % tuple(popt) )
    #plt.plot(e_array, frs, 'red', label='FRS Rate')
    #plt.plot(e_array, net/frs, 'magenta', label='Ratio of Net & FRS Rates')
    #plt.xlabel("Epsilon values")
    #plt.ylabel("Rate")
    #plt.title('T=%5.3f and T_cm=%5.3f' % tuple(temps))
    #plt.legend(loc='lower right')
    #plt.show()
    
    return A,n

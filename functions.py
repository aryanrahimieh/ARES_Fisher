#!/usr/bin/env python
# coding: utf-8

import numpy as np
import ares
from scipy import interpolate
import matplotlib.pyplot as plt
import time


def data_matrix(nu, params):
    ####################################################################
    # 1- A matrix for saving the data: M.shape= (1+3*npar, lennu)      #
    #         first row: nu, len(nu) = lennu                           #
    #         next 2: first parameter's 2 neighbors                    #
    #         next 2: second parameter's 2 neighbors ...               #
    # returns an array of interpolated dTb as a func. of nu            #
    ####################################################################
    sim = ares.simulations.Global21cm(**params, verbose=False)     # Initializing a simulation object
    sim.run()
    x = sim.history['nu']
    y = sim.history['dTb']
    f = interpolate.interp1d(x, y)
    return f(nu)


def fisher_matrix(der_mat, input_cov):
    ##################################################################
    #3- A function to calculate the Fisher elements using            #
    ### the 21-cm formula.                                           #
    ##################################################################
    fish = der_mat @ np.linalg.inv(input_cov) @ der_mat.T
    return fish

def DataDict(nu, fid_dict, step_dict):
    #################################################################
    # returns a dictionary with these rows:                         #
    # 1,2: frequency and the fiducial temp                          #
    # 2i+1,2i+2: temp(par+) and temp(par-), e.g. T(fX+-step)        #
    #################################################################
    time0 = time.time()
    DD = {}
    DD["nu"] = nu                                 #row1
    DD["Fiducial"] = data_matrix(nu, fid_dict)    #row2
    for name in fid_dict:
        parss = fid_dict.copy()
#         print(parss)
#         dlabel = name+"+"                                     # +run row 2i+1
#         parss.update({name: fid_dict[name]+step_dict[name]})
#         DD[dlabel] = data_matrix(nu, parss)
#         print(parss)
        dlabel = name+"-"                                     # -run row 2i+2
        parss.update({name: fid_dict[name]-step_dict[name]})
        DD[dlabel] = data_matrix(nu, parss)
#         print(parss, "\n")
    print("Data Dictionary Runtime = %1.2f seconds." % (time.time() - time0))
#     print(DD.keys())
    return DD

def DerivDict(datadict, fid_dict, step_dict):
    ##########################################################
    #                                                        #
    ##########################################################
    time0 = time.time()
    partial = {}
    partial["nu"] = datadict["nu"]

    for name in fid_dict:
        parss = fid_dict.copy()
#         print(parss)
#         ul = name + "+"                                        # +label
        dl = name + "-"                                        # -label
#         print(name, ul,dl,step_dict[name])
#         deriv = (datadict[ul] - datadict[dl]) / 2. / step_dict[name]
        deriv = (datadict['Fiducial'] - datadict[dl]) / step_dict[name]
        derivlog10 = deriv * np.log(10) * fid_dict[name]       # correction: derivative with respect to log10(param)
#         print(fid_dict[name])
        partial[name] = deriv
#         print("derivative finished", "\n")
    print("Derivative Dictionary Runtime = %1.8f seconds." % (time.time() - time0))
    print(partial.keys())
    return partial

def radiometer_noise(nu, T408=20, nu408=408, beta=-2.55, dnu=1000, tobs=10*3600):
    #########################################################
    # dnu in Hz, tobs in s, nu in MHz                       #
    # default values: nu408=408 MHz, beta=-2.55,            #
    # T408=20K, dnu=1KHz, tobs=10hr                         #
    #########################################################
    Tsky = T408 * (nu / nu408)**(beta)
    sigma = Tsky / np.sqrt(tobs*dnu)
    return sigma


def StepFinder(steps,  nu_arr, fid, param='fX'):
    #########################################################
    # This function is designed to find the interval of     #
    # stepsizes for each parameter with the right derivative#
    # Input: param(string)                                  #
    #########################################################
    print('steps array: ',steps,'\n parameter: ',param,'\n frequencies: ', min(nu_arr),',', max(nu_arr))
    print('fid:', fid)
    time0 = time.time()

    fid_dict = {param: fid}
    step_dict = {param: 0.0}
    stability = {}
    for i, step in enumerate(steps):
        step_dict[param] = step
        print(step_dict)
        data_dict  = DataDict(nu_arr,    fid_dict, step_dict)
        deriv_dict = DerivDict(data_dict, fid_dict, step_dict)
        if i==0:
            stability['nu'] = deriv_dict['nu']
        stability[str(step)] = deriv_dict[param]

    print("--- %s seconds ---" % (time.time() - time0))
    return stability


def Plot_der_step(stability_, param):
    #########################################################
    #                                                       #
    #########################################################
    plt.subplots(figsize=(12,6))

    for i,step in enumerate(stability_):
        if i>0:
            plt.plot(stability_['nu'], stability_[str(step)], label= param+"=" + str(step))
    label = "$\partial T / \partial $"+param
    plt.grid();plt.xlabel("$\mathcal{V}$",fontsize=15);plt.ylabel(label,fontsize=15)
    fname = "deriv_nu_"+param+".pdf"
    plt.legend(loc='upper right');plt.savefig(fname)
    return None

def residual_plot(name, fid, std, nu_arr, noise):
    #########################################################
    #                                                       #
    #########################################################
    fid_dict = {name: fid}
    step_dict = {name: std}

    data_dict_res = DataDict(nu_arr, fid_dict, step_dict)
    namep = name+"-"

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
    axes1 = plt.subplot(111)
    plt.plot(nu_arr, data_dict_res['Fiducial']-data_dict_res[namep], 'b',
    marker = 'o', markersize = 5, label="Residual = T(fid)- T(fid-1_σ)")
    plt.scatter(nu_arr, noise, marker = 'v', color='k', label="Input Covariance")
    plt.scatter(nu_arr, -noise, marker = '^', color='k')

    plt.fill_between(nu_arr, noise, 100, color='grey', alpha = 0.3)
    plt.fill_between(nu_arr, -noise, -100, color='grey', alpha = 0.3)

    txt = name + ": fid=" + str(fid) + ', 1_σ='+ str(np.round(std,5))
    plt.title(txt, fontsize = 15);#plt.ylim(-20,20)
    plt.ylabel('T(mK)',fontsize=15)
    plt.grid();plt.legend(fontsize=15)
    print('# of datapoints = ', len(nu_arr))
    chi2_value = np.sum(((data_dict_res['Fiducial']-data_dict_res[namep])/noise)**2)
    print('χ^2 = Σ(ΔTi/δi)^2 = ', np.round(chi2_value, 5))


#     axes2 = plt.subplot(212)
#     plt.plot(nu_arr, np.abs((data_dict_res['Fiducial']-data_dict_res[namep])/noise),
#     label="residual = T(fid)- T(fid-step)")
#     plt.plot(nu_arr, (data_dict_res['Fiducial']-data_dict_res[namep])/noise, "--", color='blue')
#     print(np.sum(((data_dict_res['Fiducial']-data_dict_res[namep])/noise)**2))
#     plt.plot(nu_arr, nu_arr*0, ":", color='red')
#     plt.title('delta T / radio_noise', fontsize = 15)
#     plt.ylim(-1,1)
#     plt.xlabel("$\mathcal{V}$ (MHz)",fontsize=15)

    txt = txt + ".pdf"
    plt.grid();plt.savefig(txt)
    return None

def linear_step(dict0, dict_m):
    #########################################################
    # Combines two dictionaries. Output will give the step  #
    # size in linear space.                                 #
    #########################################################
    dict3 = {}
    for key in dict_m:
    	if key in dict0:
        	dict3[key] = 10**(dict0[key])-10**(dict0[key]-dict_m[key])
    	else:
        	pass
    return dict3

def datadict_oneparam(nu, fid_dictt, p_name, par_ar):
    #################################################################
    # returns a dictionary with these rows:                         #
    # 1,2: frequency and the fiducial temp                          #
    # 2i+1,2i+2: temp(par+) and temp(par-), e.g. T(fX+-step)        #
    #################################################################
    time0 = time.time()
    DD = {}
    DD["nu"] = nu                                   #row1
    DD["Fiducial"] = data_matrix(nu, fid_dictt)    #row2

    for i in range(len(par_ar)):
#         print("i =", i)
        parss = fid_dictt.copy()
#         print("fid =", parss)
        label = p_name+str(par_ar[i])
        parss.update({p_name: par_ar[i]})
        DD[label] = data_matrix(nu, parss)
#         print("dict = ",parss, "\n")

    print("Data Dictionary Runtime = %1.2f seconds." % (time.time() - time0))
#     print(DD.keys())
    return DD

def chi2(datadict, p_name, par_ar, er):
    chi_2 = np.zeros(len(par_ar))
    for i in range(len(par_ar)):
        label = p_name+str(par_ar[i])
        chi_2[i] = np.sum(((datadict[label]-datadict['Fiducial'])/ (1000*er))**2)
    return chi_2

def plot_likelihood(datadict, p_name, par_ar, er):
    fig, ax = plt.subplots(figsize=(15,5))
    chi_2 = chi2(datadict, p_name, par_ar, er)
    plt.plot(par_ar, np.exp(-chi_2), ".", markersize =10)
    plt.xlabel(p_name, fontsize = 15); plt.ylabel("Likelihood", fontsize = 15);plt.grid();
    # ax.xaxis.set_ticks(np.arange(0.98, 1.02, 0.002));
    plt.savefig(p_name+'_Likelihood.png'); plt.close(fig)
    return None

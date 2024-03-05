import numpy as np
import matplotlib.pyplot as plt
import ares
from scipy import interpolate
import time
import h5py
import functions as f
# from fishchips.cosmo import Observables
import fishchips.util

fid_dict =  {'fesc': .1,    'Nlw': 9690.0,'fX': 1.0,  'Tmin': 10000.0}
step_dict = {'fesc': 0.003, 'Nlw': 1e0,  'fX': 1e-4, 'Tmin': 1e1}

# fid_dict =  {'fX': 1.0, 'Nlw': 9690.0}
# step_dict = {'fX': 1e-4, 'Nlw': 1e0}
nu_arr = np.linspace(50, 100, 10)

datadict = f.DataDict(nu_arr, fid_dict, step_dict)
derivdict = f.DerivDict(datadict, fid_dict, step_dict)

ordered_names = list(fid_dict.keys())
fids = list(fid_dict.values())
derivemat = np.array([derivdict[i] for i in ordered_names])
noise = f.radiometer_noise(nu_arr, T408=22.69815240532902 ,
                            nu408=408, beta=-2.564744186732979,
                            dnu=390.625e3, tobs=107*3600, Trec=300)
########uncomment to include white noise######
# noise_sys = 24e-3 # [K]
# noise = noise_r + noise_sys
##############################################
# noise = np.array([40+35, 6+35, 3+35])/1000
inputcov_mat = np.identity(len(noise)) * (1000*noise)**2
fish_mat = f.fisher_matrix(derivemat, inputcov_mat)
cov_mat = np.linalg.inv(fish_mat)

#####################
# print(fish_mat, '\n', cov_mat, '\n')
# print('sigmaNlw = %1.2e' %np.sqrt(cov_mat[1][1]))
########residual plots #############
print(cov_mat)
print('param\t fiducial \t 1-σ (Fisher)')
for i in range(len(fids)):
    print(ordered_names[i], "\t", np.round(fids[i],5), "\t",
          np.round(np.sqrt(cov_mat[i,i]),5), "\t")
# for i in range(len(fids)):
#     # print(i)
#     f.residual_plot(name=ordered_names[i], fid=fids[i], std=np.sqrt(cov_mat[i][i]),
#                     nu_arr=nu_arr, noise=1000*noise, ylim = 10)
# f.residual_plot(name=ordered_names[0], fid=fids[0], std=np.sqrt(cov_mat[0][0]),
#                 nu_arr=nu_arr, noise=1000*noise, ylim = 80)
f.residual_plot(name=ordered_names[1], fid=fids[1], std=np.sqrt(cov_mat[1][1]),
                nu_arr=nu_arr, noise=1000*noise, ylim = 40)
f.residual_plot(name=ordered_names[2], fid=fids[2], std=np.sqrt(cov_mat[2][2]),
                nu_arr=nu_arr, noise=1000*noise, ylim = 40)
f.residual_plot(name=ordered_names[3], fid=fids[3], std=np.sqrt(cov_mat[3][3]),
                nu_arr=nu_arr, noise=1000*noise, ylim = 40)
########################

fig, axes = fishchips.util.plot_triangle_base(ordered_names, fids, cov_mat, labels=ordered_names,
                                             fig_kwargs={'figsize': (15, 15)},
                                             xlabel_kwargs={'labelpad': 30, 'fontsize':30},
                                             ylabel_kwargs={'labelpad': 30, 'fontsize':30});

l1, = axes[0, -1].plot([],[],'-',color="black", label='radio noises 10dps EDGES range')
axes[0, -1].legend(fontsize="20")
plt.savefig('triangle_rn_10_EDGES_range.png')

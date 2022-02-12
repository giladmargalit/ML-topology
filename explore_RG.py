# -*- coding: utf-8 -*-

import numpy as np
import keras
from keras.models import *
from matplotlib import pyplot as plt
import tensorflow as tf

from custom_functions import *

keras.losses.custom_loss = custom_loss

def get_RG_4x4(mu, t_minus, Delta):
    cur_input = np.concatenate(( mu,t_minus,Delta ))
    cur_input_reshaped = np.reshape(cur_input,[1,N_features])
    out_4x4 = model_8x8.predict(cur_input_reshaped)
    return out_4x4[1][0]

def get_general_RG(model_8x8, N_orig):
    input_size_orig = (N_orig**2)*3
    RG_layer_model = Model(inputs=model_8x8.input,
                                       outputs=model_8x8.get_layer('final_4x4').output)
    inputs_NxN = Input(shape=(input_size_orig,), dtype='float32')
    x = RG_layer_model(inputs_NxN)
    model_RG = Model(inputs=inputs_NxN, outputs=x)
    return model_RG

def predict_general_RG(model_RG, mu, t_minus, Delta):
    cur_input = np.concatenate(( mu, t_minus, Delta ))
    input_size_orig = len(cur_input)
    N_orig = np.sqrt(input_size_orig/3).astype(int)
    cur_input_reshaped = np.reshape(cur_input, [1, input_size_orig])
    pred = model_RG.predict(cur_input_reshaped)
    return pred[0]

def draw_lattice(data, title='', cmaps=['Reds', 'Greens', 'Blues'], shrink_factor=0.22):
    fig, ax = plt.subplots(1, 3, dpi=300)
    params = [r'$\mu$', r'$t_{-}$', r'$\Delta$']
    for j in range(3):
        cur_img = ax[j].imshow(data[:, :, j], origin='lower', cmap=cmaps[j],
               extent=(0, 4, 0, 4))
        ax[j].set_title(params[j])
        cbar = plt.colorbar(mappable=cur_img, ax=ax[j], shrink=shrink_factor, aspect=25*shrink_factor)
        ax[j].set_xticks([]); ax[j].set_yticks([])
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=1.4) 
    return fig, ax

def draw_difference_lattice(data, data_orig, title='',
                            cmaps=['Reds', 'Greens', 'Blues'], shrink_factor=0.22):
    fig, ax = plt.subplots(1, 3, dpi=300)
    params = [r'$\mu$', r'$t_{-}$', r'$\Delta$']
    for j in range(3):
        cur_img = ax[j].imshow(data[:, :, j]-data_orig[:, :, j],
                               origin='lower', cmap=cmaps[j], extent=(0, 4, 0, 4))
        ax[j].set_title(params[j] + '-' + params[j] + r'$_{\rm orig}$')
        cbar = plt.colorbar(mappable=cur_img, ax=ax[j], shrink=shrink_factor, aspect=25*shrink_factor)
        ax[j].set_xticks([]); ax[j].set_yticks([])
    plt.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(top=1.4) 
    return fig, ax


Nx = 8
Ny = Nx

N_sites = Nx*Ny

N_labels = 3
N_channels = 3
N_features = N_channels*Nx*Ny

# fixed parameters
t_plus = 1
Delta_0 = 0.15

# load the 4x4 model and the 8x8 model, which has a 8x8->4x4 RG block
model_4x4 = load_model('4x4_model.h5')
model_8x8 = load_model('8x8_model.h5')



#%% Domain wall in mu
if 1:
    t_minus_0 = 1
    mu_1 = 1
    mu_2 = mu_1 + (-2)
    mu = np.kron([mu_1, mu_2], np.ones(Nx*Ny//2))
    t_minus = t_plus*t_minus_0*np.ones(Nx*Ny)
    Delta = t_plus*Delta_0*np.ones(Nx*Ny)
    out_4x4 = get_RG_4x4(mu, t_minus, Delta)
    draw_lattice(out_4x4, title=r'Domain wall: $\mu_{\rm top}=%.01f$, $\mu_{\rm bot}=%.01f$' % (
        mu_1, mu_2))
       
#%% Single impurity
if 1:
    t_minus_0 = 1
    mu_0 = 0.5
    mu_imp_loc = 29
    mu_imp_amp = 1
    
    mu = mu_0*np.ones(Nx*Ny)
    t_minus = t_plus*t_minus_0*np.ones(Nx*Ny)
    Delta = t_plus*Delta_0*np.ones(Nx*Ny)
    out_4x4_orig = get_RG_4x4(mu, t_minus, Delta)
    mu[mu_imp_loc] += mu_imp_amp
    out_4x4 = get_RG_4x4(mu, t_minus, Delta)
    draw_difference_lattice(out_4x4, out_4x4_orig,
                            title=r'$\mu_0=%.01f$, $t_{-,0}=%.01f$, $\Delta_0=%.02f$ | $\delta\mu=%.01f$, imp. loc. = (%d,%d)' %
              (mu_0, t_minus_0, Delta_0, mu_imp_amp, mu_imp_loc//Nx, np.mod(mu_imp_loc, Nx)))

#%% Multiple impurities
if 1:
    t_minus_0 = 1
    mu_0 = 0.5
    mu_imp_locs = [0, 29]
    mu_imp_amp = 0.3
    
    mu = mu_0*np.ones(Nx*Ny)
    for mu_imp_loc in mu_imp_locs:
        mu[mu_imp_loc] += mu_imp_amp
    t_minus = t_plus*t_minus_0*np.ones(Nx*Ny)
    Delta = t_plus*Delta_0*np.ones(Nx*Ny)
    out_4x4 = get_RG_4x4(mu, t_minus, Delta)
    draw_lattice(out_4x4, title=r'$\mu_0=%.01f$, $t_{-,0}=%.01f$, $\Delta_0=%.02f$ | %d impurities, $\delta\mu=%.01f$' %
              (mu_0, t_minus_0, Delta_0, len(mu_imp_locs), mu_imp_amp))

   
#%% NxN: single impurity
if 1:
    N_orig = 32
    model_RG = get_general_RG(model_8x8, N_orig)
    
    t_minus_0 = 1
    mu_0 = 0.5
    mu_imp_loc = int(0.6*N_orig**2)
    mu_imp_amp = 0.01
    
    mu = mu_0*np.ones(N_orig**2)
    mu[mu_imp_loc] += mu_imp_amp
    t_minus = t_plus*t_minus_0*np.ones(N_orig**2)
    Delta = t_plus*Delta_0*np.ones(N_orig**2)
    
    data_RG = predict_general_RG(model_RG, mu, t_minus, Delta)
    draw_lattice(data_RG, title=r'$\mu_0=%.01f$, $t_{-,0}=%.01f$, $\Delta_0=%.02f$ | $N=%d$' %
              (mu_0, t_minus_0, Delta_0, N_orig))
    

#%% Timing analysis
if 1:
    N_vec = [4, 8, 16, 32, 64, 128]
    upto_N_log = int(np.log2(N_vec[-1])) - 3
    model_RG = upto_N_log*[[]]
    for n in range(upto_N_log):
        model_RG[n] = get_general_RG(model_8x8, 2**(4+n))
    
    print('N\tavg. time')
    print('-------------')
    avg_time_vec = np.zeros(len(N_vec))
    
    for jn, Nx in enumerate(N_vec):
        Ny = Nx
        num_re = 50
        tic = tics()   
        for re in range(num_re):
            in_data = np.reshape(np.random.randn(Nx*Ny*3), [1, Nx*Ny*3])    
            if Nx == 4:
                out_data = model_4x4.predict(in_data)
            elif Nx == 8:
                out_data = model_8x8.predict(in_data)
            else:
                N = Nx
                while N > 8:
                    out_data = model_RG[int(np.log2(N))-4].predict(in_data)
                    N = int(N/2)
                    in_data = np.reshape(out_data[0].flatten(), [1, N*N*3])
                out_data = model_8x8.predict(in_data)
        
        avg_time = toc(tic, disp=False)/num_re
        avg_time_vec[jn] = avg_time
        
        print('%d\t%.04f' % (Nx, avg_time))
           
    plt.figure(dpi=200)
    plt.scatter(np.log2(N_vec), np.log(avg_time_vec/avg_time_vec[0]))
    plt.xticks(np.log2(N_vec), labels=N_vec)
    plt.xlabel(r'Linear lattice size $N$')
    plt.ylabel('log of normalized time')
    
#%% Points mapping
if 1:
    # Fixed parameters
    t_plus = 1
    Delta_0 = 0.15
    # Plotting parameters
    L = 2.1; s=0.1; lw=1
    
    xi_vec = np.linspace(-2, 2, 100)
    eta_vec = np.linspace(-2, 2, 100)
    
    mu_dis = 0*np.random.randn(8*8)
    t_minus_dis = 0*np.random.randn(8*8)
    Delta_dis = 0*np.random.randn(8*8)
    
    new_points_xi = np.zeros([len(xi_vec), len(eta_vec)])
    new_points_eta = np.zeros([len(xi_vec), len(eta_vec)])
    
    for xi_idx in range(len(xi_vec)):
        for eta_idx in range(len(xi_vec)):
            mu_0 = 2*xi_vec[xi_idx]
            mu = t_plus * (mu_0 + mu_dis)
            
            t_minus_0 = eta_vec[eta_idx]
            t_minus = t_plus * (t_minus_0 + t_minus_dis)
            
            Delta = t_plus * (Delta_0 + Delta_dis)
    
            cur_input = np.concatenate(( mu, t_minus, Delta ))
            
            intermediate_output = model_8x8.predict(
                np.reshape(cur_input, [1, len(cur_input)]))[1]
            new_mu = np.mean(intermediate_output[0,:,:,0])
            new_t_minus = np.mean(intermediate_output[0,:,:,1])
            new_Delta = np.mean(intermediate_output[0,:,:,2])
            new_points_xi[xi_idx, eta_idx] = new_mu / (2*t_plus)
            new_points_eta[xi_idx, eta_idx] = new_t_minus/t_plus

    plt.figure(dpi=200)
    plt.scatter(new_points_xi, new_points_eta, s=0.1)
    plt.xlabel(r'$\mu/$2$t^{+}$')
    plt.ylabel(r'$t^{-}/t^{+}$')
    plt.xticks([-2, 0, 2]); plt.yticks([-2, 0, 2])
    plt.plot([-L, L], [-L, L], 'k', lw=lw)
    plt.plot([-L, L], [L, -L], 'k', lw=lw)
    plt.plot([-1, -1], [-L, L], 'k', lw=lw)
    plt.plot([1, 1], [-L, L], 'k', lw=lw)
    plt.xlim(-L, L)
    plt.ylim(-L, L)


        


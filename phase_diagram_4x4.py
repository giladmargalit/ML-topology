# -*- coding: utf-8 -*-

from custom_functions import *
from scipy.io import loadmat
from matplotlib import gridspec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def plot_phase_diagram(xi_vec, eta_vec, chern_mat, cmap='', aspect=1):
    chern_mat[np.where(chern_mat>1)] = 1
    chern_mat[np.where(chern_mat<-1)] = -1
    if cmap == '': # create custom colormap
        newcolors = (1/256)*np.array([[202, 0, 32],
                                      [247, 247, 247],
                                      [5, 113, 176]])
        cmap = ListedColormap(newcolors)
    plt.imshow(np.flipud(np.transpose(chern_mat)), origin='lower',
                     cmap=cmap, aspect=aspect,
            extent=[min(xi_vec),max(xi_vec),min(eta_vec),max(eta_vec)])
    plt.xlabel(r'$\mu/2t_{+}$')
    plt.ylabel(r'$t_{-}/t_{+}$')
    cbar = plt.colorbar(ticks=[-0.75, 0, 0.75])
    cbar.ax.set_yticklabels(['-1', '0', '1'])
    
    

# Load the trained 4x4 model
model = load_model('4x4_model.h5')

# Choose 4x4 example (1, 2, 3) to plot its phase diagram
data_dir = 'examples_4x4'
example_idx = 1


t_plus = 1
Delta_0 = 0.15
data_file = data_dir + '/data_' + str(example_idx) + '.mat'
data = loadmat(data_file)

xi_vec = data['xvec'].flatten()
eta_vec = data['yvec'].flatten()
chern_mat = np.zeros([len(xi_vec), len(eta_vec)])

"""
This code can be used for a general disorder configuration: just set the variables
mu_dis, t_minus_dis, Delta_dis to the desired configuration.
"""
mu_dis = data['mu_dis'].flatten()
t_minus_dis = data['t_minus_dis'].flatten()
Delta_dis = data['Delta_dis'].flatten()

for xi_idx in range(len(xi_vec)):
    for eta_idx in range(len(xi_vec)):
        mu_0 = 2*xi_vec[xi_idx]
        mu = t_plus * (mu_0 + mu_dis)
        
        t_minus_0 = eta_vec[eta_idx]
        t_minus = t_plus * (t_minus_0 + t_minus_dis)
        
        Delta = t_plus * (Delta_0 + Delta_dis)

        cur_input = np.concatenate(( mu, t_minus, Delta ))
        
        chern_mat[xi_idx,eta_idx] = model.predict(np.reshape(
            cur_input, [1, len(cur_input)]))[0]


# Plot the phase diagram
plt.figure(dpi=200)
plot_phase_diagram(xi_vec, eta_vec, chern_mat)

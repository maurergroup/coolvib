#    This file is part of coolvib
#
#        coolvib is free software: you can redistribute it and/or modify
#        it under the terms of the GNU General Public License as published by
#        the Free Software Foundation, either version 3 of the License, or
#        (at your option) any later version.
#
#        coolvib is distributed in the hope that it will be useful,
#        but WITHOUT ANY WARRANTY; without even the implied warranty of
#        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#        GNU General Public License for more details.
#
#        You should have received a copy of the GNU General Public License
#        along with coolvib.  If not, see <http://www.gnu.org/licenses/>.
"""
We calculate the (6x6) friction tensor for 
a CO molecule adsorbed on a Pd(111) top site.
"""

import numpy as np
from ase.io import read
import os, sys
import coolvib
from scipy import linalg as LA
#from ase.visualize import view

#####DEFINE SYSTEM and PARAMETERS######
system = read('CO_on_Pd111/eq/geometry.in')
cell = system
active_atoms = [4,5] # meaning we have two atoms - C - 0 and O - 1
#view(system)

model = coolvib.workflow_tensor(system, code='aims', active_atoms=active_atoms)

finite_difference_incr = 0.0025

keywords = {
    'discretization_type' : 'gaussian',
    'discretization_broadening' : 0.01,
    'discretization_length' : 0.01,
    'max_energy' : 3.00,
    'temperature' : 300,
    'delta_function_type': 'gaussian',
    'delta_function_width': 0.60,
    'perturbing_energy' : 0.0,
    'debug': 0,
        }

print('workflow initialized and keywords set')

######READ QM INPUT DATA###
model.read_input_data(
        spin=False, 
        path='./CO_on_Pd111/',
        filename='aims.out', 
        active_atoms=active_atoms, 
        incr=finite_difference_incr,
        debug=0,
        )
print('successfully read QM input data')

############
#We can either calculate the spectral function and evaluate friction 
#from there (discretization commands)

######CALCULATE SPECTRAL FUNCTION###
# model.calculate_spectral_function(mode='default', **keywords)
# # model.read_spectral_function()
# print 'successfully calculated spectral_function'
# model.print_spectral_function('nacs-spectrum.out')
# model.calculate_friction_tensor_from_spectrum(**keywords)
# model.plot_spectral_function()

#######CALCULATE FRICTION TENSOR###

#or we directly calculate friction
model.calculate_friction_tensor( **keywords)

print('successfully calculated friction tensor')

model.analyse_friction_tensor()
model.print_jmol_friction_eigenvectors()


######project lifetimes along vibrations
hessian_modes = np.loadtxt('CO_on_Pd111/hessian')
for i in range(len(hessian_modes)):
    hessian_modes[i,:]/=np.linalg.norm(hessian_modes[i,:])

from coolvib.constants import time_to_ps
ft= np.dot(hessian_modes,np.dot(model.friction_tensor,hessian_modes.transpose()))/time_to_ps
E, V = np.linalg.eigh(ft)
coolvib.routines.output.print_matrix(ft)

print('Relaxation rates along vibrational modes in 1/ps')
for i in range(len(ft)):
    print(ft[i,i])
print('Vibrational lifetimes along vibrational modes in ps')
for i in range(len(ft)):
    print(1./ft[i,i])

######Print vibrational lifetimes and relaxation rates to a file instead######   
#file_output = 'relax_rate_CO_on_Pd111.txt'    
#with open(file_output, 'w') as f:
#    f.write("Relaxation rates along vibrational modes in 1/ps \n")
#    for i in range(len(ft)):
#        f.write(str(ft[i,i]))
#        f.write("\n")
#    f.write("Vibrational lifetimes along vibrational modes in ps \n")
#    for i in range(len(ft)):
#        f.write(str(1./ft[i,i]))
#        f.write("\n")

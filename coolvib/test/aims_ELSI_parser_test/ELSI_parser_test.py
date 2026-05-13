
import numpy as np
import coolvib
from coolvib.routines.friction_tensor import *
from coolvib.routines.spectral_function import *
from ase.io import read
from ase import *

system = read('./test_data/geometry.in')
cell = system
active_atoms = [2]
model = coolvib.workflow_tensor(system, code='aims', active_atoms=active_atoms)

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

finite_difference_incr = 0.0025

model.read_input_data(
        spin=False, 
        path='./test_data/',
        filename='aims.out', 
        active_atoms=active_atoms, 
        incr=finite_difference_incr,
        debug=0,
)

model.calculate_spectral_function(mode='default', **keywords)
model.read_spectral_function()
print('successfully calculated spectral_function')
model.print_spectral_function('nacs-spectrum.out')
model.calculate_friction_tensor_from_spectrum(**keywords)
model.plot_spectral_function()

""" 
spectral_function = calculate_spectral_function_tensor_from_elph_matrix(
        el_ph_coupling_path = "./test_data/",
        fermi_energy = model.fermi_energy,
        eigenvalues = model.eigenvalues,
        kpoints = model.kpoints,
        psi = model.psi,
        masses = model.atoms.get_masses()[self.active_atoms],
        ** keywords)
"""
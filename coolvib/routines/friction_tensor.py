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
friction_tensor.py
"""

from coolvib.routines import evaluate_delta_function
from coolvib.routines.output import print_matrix
from coolvib.constants import time_to_ps

import numpy as np
import numpy.linalg as la

def calculate_tensor(n_dim, x_axis, spectral_function, **kwargs):
    """
    calculates friction tensor from given spectral function
    """
    
    ##ANALYSE KWARGS

    default_keywords = {
            'delta_function_type' : 'gaussian',
            'delta_function_width' : 0.60,
            'max_energy' : 3.0,
            'perturbing_energy' : 0.0,  #centered around fermi
            }

    keys = {}
    for key in default_keywords.keys():
        if key in kwargs:
            keys[key] = kwargs[key] 
        else:
            keys[key] = default_keywords[key]

    sigma = keys['delta_function_width']
    delta_method = keys['delta_function_type']
    perturbing_energy = keys['perturbing_energy']
    friction_tensor = np.zeros([n_dim,n_dim],dtype=np.complex)
    
    c = 0
    for d in range(n_dim):
        for d2 in range(d,n_dim):
            
            friction_tensor[d,d2] = \
                    np.real(evaluate_delta_function(x_axis, spectral_function[c], perturbing_energy ,sigma, delta_method))
            friction_tensor[d2,d] = np.real(friction_tensor[d,d2].conjugate())

            c += 1

    return friction_tensor 


def analyse_tensor(friction_tensor):
    """
    performs a spectral analysis of friction tensor
    """

    eigenvalues, eigenvectors = la.eigh(friction_tensor)

    #printing
    print('Friction Tensor') 
    print_matrix(friction_tensor/time_to_ps)
    print(' ')
    print('Friction Eigenvalues in 1/ps')
    print(' '.join( ['{0:10d}'.format(y) for y in range(len(eigenvalues)) ]))
    print(' '.join( ['{0:10.4f}'.format(y) for y in eigenvalues/time_to_ps ]  ))
    print('Friction Eigenvectors')
    print_matrix(eigenvectors)
    print('Principal lifetimes in ps')
    print(' '.join( ['{0:10d}'.format(y) for y in range(len(eigenvalues)) ]))
    print(' '.join( ['{0:10.4f}'.format(y) for y in 1./(eigenvalues/time_to_ps) ]  ))

    return eigenvectors, eigenvalues
    
    #do stuff

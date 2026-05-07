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
test_aims_parsing.py

tests the different routines in parser/aims.py
"""

import numpy as np
from coolvib.parser.aims import *
from ase.io import read
from ase import *

atoms = read('aims_parser_test/geometry.in')
cell = atoms.cell

filename = 'aims_parser_test/OUTPUT'

fermi_level, kpoint_weights = aims_read_fermi_and_kpoints(filename, cell)

print(fermi_level)

print(kpoint_weights)

nkpts = len(kpoint_weights)

eigenvalues, psi, occ, orb_pos = aims_read_eigenvalues_and_coefficients(fermi_level, './aims_parser_test/', spin=True)

print(eigenvalues)
print(eigenvalues.shape)
print(psi.shape)

H, S = aims_read_HS('./aims_parser_test/',spin=True)

print(H.shape)

print(S.shape)

nspin = 0
nk = 0

psi1 = psi[nk,nspin,:,:]
S1 = S[nk,:,:]
H1 = H[nk,nspin,:,:]
S_inv = np.linalg.inv(S1)
S_invH = np.dot(S_inv,H1)

import scipy.linalg as la
E, V = la.eigh(H1,S1)
print(E*27.2114)

for i in range(len(psi1)):
    print(np.dot(psi1[i].conjugate(),np.dot(S_invH,psi1[i]))/np.dot(psi1[i].conjugate(),psi1[i])*27.2114)


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

from sys import exit

if __name__=="__main__":
    pass
else:
    exit()

from coolvib.parser.siesta import *

filename = 'siesta_parser_test/MyM'

cell = siesta_read_struct_out(filename)
print(cell)

fermi_level, eigenvalues = siesta_read_eigenvalues(filename)

print('fermi_level ', fermi_level)
print('eigenvalue shape ',eigenvalues.shape)

kpt_array = siesta_read_kpoints(filename)

print('kpts')
print(kpt_array.shape)
#print kpt_array

psi, orbital_pos =siesta_read_coefficients(filename)

print('psi')
print(psi.shape, orbital_pos)

from time import time
start = time()
#kpt_array = kpt_array[:5]
#print kpt_array
H,S = siesta_read_HSX(kpt_array, filename,debug=1)
print('time ', time() - start)

print('H')
print(H.shape)
print('S')
print(S.shape)

nspin = 0
e1 = []
e2 = []
#for nk in range(len(kpt_array)):
for nk in [0]:
    print(nk)
    psi1 = psi[nk,nspin,:,:]
    psi1_cg = psi.conjugate()[nk,nspin,:,:]
    H1 = H[nk,nspin,:,:]
    S1 = S[nk,:,:]
    S1_inv = np.linalg.inv(S1)
    e = eigenvalues[nk,nspin,:]
    print(e[0])
    e1.append(e[0])
    import scipy.linalg as la
    E, V = la.eigh(H1,S1)
    print(E[0])
    e2.append(E[0])

#print sorted(e1)
#print sorted(e2)
#S1_invH = np.dot(S1_inv,H1)

#for i in range(len(psi[nk,nspin,:,0])):
    #print np.dot(psi1_cg[i],np.dot(S1_invH,psi1[i]))#/np.dot(psi1[i].conjugate(),psi1[i])

#print np.dot(psi1.conjugate().transpose(),np.dot(S1_invH,psi1))/np.dot(psi1.conjugate().transpose(),psi1.conjugate())
print(np.dot(psi1.conjugate().transpose(),np.dot(H1,psi1)))
print(e*np.dot(psi1.conjugate().transpose(),np.dot(S1,psi1)))


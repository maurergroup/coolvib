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
build_G.pyx is a Cython extension to the coolvib.routines.spectral_function module.
This is very time consuming and therefore written in Cython.

"""

import cython
import numpy as np
cimport numpy as np
REAL_TYPE = np.float
INT_TYPE = np.int
COMPLEX_TYPE = np.complex
ctypedef np.int_t INT_TYPE_t
ctypedef np.float_t REAL_TYPE_t
ctypedef np.complex_t COMPLEX_TYPE_t

from libc.math cimport sin, cos
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def siesta_calc_HSX(int nspin, np.ndarray[REAL_TYPE_t,ndim=2] kpts_array, 
                    int no_u, np.ndarray[INT_TYPE_t,ndim=1] numh, 
                    np.ndarray[INT_TYPE_t,ndim=1] listhptr,
                    np.ndarray[INT_TYPE_t,ndim=1] listh, 
                    np.ndarray[INT_TYPE_t,ndim=1] indxuo,
                    np.ndarray[REAL_TYPE_t,ndim=2] xij,
                    np.ndarray[REAL_TYPE_t,ndim=2] h,
                    np.ndarray[REAL_TYPE_t,ndim=1] s,
            ):

    raise NotImplementedError('This is work in progress')
   

    real_H_r = np.zeros([n_atoms,n_cart,nk,nk,ns,n_basis,n_basis],dtype=np.complex) 
    real_S_r = np.zeros([n_atoms,n_cart,nk,nk,n_basis,n_basis],dtype=np.complex) 
    for N1 in range(nk):
        Nvec1 = N[N1] 
        for N2 in range(nk):
            Nvec2 = N[N2]
            print 'working on N1 and N2  ', N1, N2
            for l in range(nk):
                kvec = kpts[l]
                kw = kweights[l]
                kx = kvec[0]*(Nvec2[0]-Nvec1[0])+
                     kvec[1]*(Nvec2[1]-Nvec1[1])+
                     kvec[2]*(Nvec2[2]-Nvec1[2])
                phase = np.exp(1.0j*kx)             
                real_H_r[:,:,N1,N2,:,:,:]+=first_order_H[:,:,l,:,:,:]*phase*kw
                real_S_r[:,:,N1,N2,:,:]+=first_order_S[:,:,l,:,:]*phase*kw
    #print 'Built real space first_order_H and first_order_S'
   
    G = np.zeros([n_atoms,n_cart,nk,nk,ns,n_basis,n_basis],dtype=np.complex)
   
    counter = 0
    for atom in range(n_atoms):
        for cart in range(n_cart):
    #print 'calculating G for atom {0} and cart {1}'.format(atom,cart)
            for s in range(ns):
                for k1 in range(nk):
                    print 'k1 ', k1
                    kvec1 = kpts[k1]
                    for k2 in range(nk):
                        print 'k2 ', k2
                        kvec2 = kpts[k2]
                        for N1 in range(nk):
                            Nvec1 = N[N1] 
                            for N2 in range(nk):
                                Nvec2 = N[N2]
                                tmpH = np.zeros([n_atoms,n_cart,n_basis,n_basis],dtype=np.complex)
                                tmpS = np.zeros([n_atoms,n_cart,n_basis,n_basis],dtype=np.complex)
                                for l in range(nk):
                                    kvec = kpts[l]
                                    kw = kweights[l]
                                    kx = kvec[0]*(Nvec2[0]-Nvec1[0])+
                                         kvec[1]*(Nvec2[1]-Nvec1[1])+
                                         kvec[2]*(Nvec2[2]-Nvec1[2])
                                    phase = np.exp(1.0j*kx)             
                                    tmpH += first_order_H[:,:,l,s,:,:]*phase*kw
                                    tmpS += first_order_S[:,:,l,:,:]*phase*kw

                                kx = kvec2[0]*Nvec2[0]-kvec1[0]*Nvec1[0]+
                                     kvec2[1]*Nvec2[1]-kvec1[1]*Nvec1[1]+
                                     kvec2[2]*Nvec2[2]-kvec1[2]*Nvec1[2]
                                phase = np.exp(1.0j*kx)
                                G[:,:,k1,k2,s,:,:] += (tmpH-fermi_energy*tmpS)*phase


    return G


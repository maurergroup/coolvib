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
siesta_mod.pyx is a Cython extension to the coolvib.parser.siesta module 
that deals with the transformation of Hamiltonian and overlap matrix into 
k space. This is very time consuming and therefore written in Cython.

"""

import cython
import numpy as np
cimport numpy as np
REAL_TYPE = np.float64
INT_TYPE = np.int64
COMPLEX_TYPE = np.complex128
ctypedef np.int64_t INT_TYPE_t
ctypedef np.float64_t REAL_TYPE_t
ctypedef np.complex128_t COMPLEX_TYPE_t

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
    """
    efficiently calculates actual H and S matrices for each kpoint and spin 
    from the sparse h and s matrices using the 'Siesta approach'
    
    Parameters:



    Output:

    
    """

    #INIT H and S
    cdef int nkpts
    nkpts = len(kpts_array)
    cdef np.ndarray[REAL_TYPE_t,ndim=4] H_r = np.zeros([nkpts,nspin,no_u,no_u],dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=4] S_r = np.zeros([nkpts,nspin,no_u,no_u],dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=4] H_i = np.zeros([nkpts,nspin,no_u,no_u],dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=4] S_i = np.zeros([nkpts,nspin,no_u,no_u],dtype=REAL_TYPE)

    cdef unsigned int si, ki, iuo, j, io, jo, juo, ind
    cdef np.ndarray[REAL_TYPE_t,ndim=1] k=np.empty(4,dtype=REAL_TYPE), kvec=np.empty(3,dtype=REAL_TYPE)
    cdef REAL_TYPE_t phasef_r, phasef_i, kx

    #numh number of non-zero elements per row
    #listhptr points to the start of each orbital row
    #listh non-zero H elements for each orbital 
    #indxuo tells me which orbital in supercell corresponds
    #to which orbital in unit cell

    for ki,k in enumerate(kpts_array):
        kvec = k[:3]
        for si in range(nspin):
            for iuo in range(no_u):
                for j in range(numh[iuo]):
                    ind = listhptr[iuo] + j
                    jo = listh[ind] -1
                    juo = indxuo[jo] -1
                    kx = np.dot(kvec,xij[ind,:])
                    phasef_r = cos(kx)
                    phasef_i = sin(kx)
                    H_r[ki,si,iuo,juo] += phasef_r*h[ind,si]
                    H_i[ki,si,iuo,juo] += phasef_i*h[ind,si]
                    S_r[ki,si,iuo,juo] += phasef_r*s[ind]
                    S_i[ki,si,iuo,juo] += phasef_i*s[ind]
     
    return H_r, H_i, S_r[:,0,:,:], S_i[:,0,:,:]


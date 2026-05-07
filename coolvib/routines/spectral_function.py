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
spectral_function.py
"""

import numpy as np
from math import sqrt, pi

from coolvib.routines import fermi_occ
from coolvib.routines import delta_function 
from coolvib.routines import discretize_peak
from coolvib.constants import hplanck, hbar, time_to_ps

import coolvib.routines.build_G as build_G 

def calculate_spectral_function_mode(
        fermi_energy,
        eigenvalues,
        kpoints,
        psi,
        first_order_H,
        first_order_S,
        **kwargs):
    """
    Calculates spectral functions for all cartesian directions and 
    couplings between directions.

    Parameters:

    fermi_energy: float
        Fermi Level in eV

    eigenvalues: np.array
        numpy array of eigenvalues with dimensions given by 
        n_kpoints, n_spin, n_states

    kpoints: np.array
        array of kpoints with shape (n_kpoints, 4) where the 
        first three columns give the x, y and z components of 
        the kvector in reciprocal space and the fourth column 
        gives the kpoint weight.

    psi: np.array
        array of wavefunction coefficients with its shape given 
        by n_kpoints, n_spin, n_states and n_basisfunctions

    first_order_H: np.array
        cartesian derivative of the hamiltonian matrix with its 
        6 dimension given by n_atoms, 3 cartesian directions, 
        n_kpoints, n_spin, n_basis, n_basis

    first_order_S: np.array
        cartesian derivative of the overlap matrix with its 
        6 dimension given by n_atoms, 3 cartesian directions, 
        n_kpoints, n_spin, n_basis, n_basis
        
    Returns:

    x_axis: np.array
        array with the discretized x axis values for the spectral function

    spectral_function:: np.array
        2-dimensional array 

    """

    #ANALYSE KWARGS

    default_keywords = {
            'discretization_type' : 'gaussian',
            'discretization_broadening' : 0.01,
            'discretization_length' : 0.01,
            'max_energy' : 3.0,
            'temperature': 300,
            'debug' : 0,
            }

    keys = {}
    for key in default_keywords.keys():
        if  key in kwargs:
            keys[key] = kwargs[key] 
        else:
            keys[key] = default_keywords[key]

    #build coupling matrix G
    
    n_kpts, n_spin, n_states, n_basis = first_order_H.shape
    kweights = kpoints[:,-1]

    G = np.zeros([n_kpts,n_spin,n_states, n_basis],dtype=np.complex)
    for s in range(n_spin):
        G[:,s,:,:] = first_order_H[:,s,:,:] - fermi_energy*first_order_S[:,:,:]

    # calculate spectral components

    n_axis = int(keys['max_energy']/keys['discretization_length'])
    x_axis = np.array( [keys['discretization_length']*i for i in range(n_axis)] )
    ef = fermi_energy
    delta_method = keys['discretization_type']
    sigma = keys['discretization_broadening']
    T = keys['temperature']
    debug = keys['debug']

    spectral_function = np.zeros([1,n_axis],dtype=np.complex128)

    print('Calculating spectral function') 
    for s in range(n_spin):
        for k in range(n_kpts):
            if debug:
                print('s ', s, 'k ', k)
            wk = kweights[k]
            orb_min = 0
            orb_lumo = 0
            orb_homo = 0
            orb_max = 0
            for ei,e in enumerate(eigenvalues[k,s,:]):
                occ = fermi_occ(e,ef,T)*(2./n_spin)
                if e<=ef-2.00*keys['max_energy']:
                    orb_min = ei
                if occ>=0.999:
                    orb_lumo = ei
                if occ>= 0.001:
                    orb_homo = ei
                if e<=ef+2.00*keys['max_energy']:
                    orb_max = ei
            if debug:
                print(orb_min, orb_homo, orb_lumo, orb_max)
            for i in range(orb_min,orb_homo+1):
                for f in range(orb_lumo, orb_max+1):
                    e = eigenvalues[k,s,f] - eigenvalues[k,s,i]
                    occ =(fermi_occ(eigenvalues[k,s,i],ef,T) - fermi_occ(eigenvalues[k,s,f],ef,T))*(2./n_spin)
                    if e>0.0 and e<=1.0*keys['max_energy'] and occ>=1.E-5:
                        #calculate transition strength
                        nacs1 = np.dot(psi[k,s,i,:].conjugate().transpose(),np.dot(G[k,s,:,:],
                                psi[k,s,f,:]))
                        nacs2 = np.dot(psi[k,s,i,:].conjugate().transpose(),np.dot(G[k,s,:,:],
                                psi[k,s,f,:]))
                        nacs = np.dot(nacs1.conjugate().transpose(),nacs2)
                        nacs /= (e)
                        nacs *= wk
                        nacs *= occ
                        if debug:
                            print(i, f, e, ' ' , occ, ' ', 
                                (nacs*hbar*pi/(time_to_ps)).real, 
                                ' ', (nacs*hbar*pi/(time_to_ps)).imag)
                        spectral_function[0,:] += discretize_peak(e,nacs, x_axis, sigma, delta_method)
            
            spectral_function[0,:] *= (pi*hbar)

    return x_axis, spectral_function


def calculate_spectral_function_tensor(
        fermi_energy,
        eigenvalues,
        kpoints,
        psi,
        first_order_H,
        first_order_S,
        masses,
        **kwargs):
    """
    Calculates spectral functions for all cartesian directions and 
    couplings between directions.

    Parameters:

    fermi_energy: float
        Fermi Level in eV

    eigenvalues: np.array
        numpy array of eigenvalues with dimensions given by 
        n_kpoints, n_spin, n_states

    kpoints: np.array
        array of kpoints with shape (n_kpoints, 4) where the 
        first three columns give the x, y and z components of 
        the kvector in reciprocal space and the fourth column 
        gives the kpoint weight.

    psi: np.array
        array of wavefunction coefficients with its shape given 
        by n_kpoints, n_spin, n_states and n_basisfunctions

    first_order_H: np.array
        cartesian derivative of the hamiltonian matrix with its 
        6 dimension given by n_atoms, 3 cartesian directions, 
        n_kpoints, n_spin, n_basis, n_basis

    first_order_S: np.array
        cartesian derivative of the overlap matrix with its 
        6 dimension given by n_atoms, 3 cartesian directions, 
        n_kpoints, n_spin, n_basis, n_basis
        
    masses: np.array or list
        list of masses for the n_atoms active atoms


    Returns:

    x_axis: np.array
        array with the discretized x axis values for the spectral function

    spectral_function:: np.array
        2-dimensional array 

    """

    #ANALYSE KWARGS

    default_keywords = {
            'discretization_type' : 'gaussian',
            'discretization_broadening' : 0.01,
            'discretization_length' : 0.01,
            'max_energy' : 3.0,
            'temperature': 300,
            'debug' : 0,
            }

    keys = {}
    for key in default_keywords.keys():
        if  key in kwargs:
            keys[key] = kwargs[key] 
        else:
            keys[key] = default_keywords[key]

    #build coupling matrix G
    
    n_atoms, n_cart, n_kpts, n_spin, n_states, n_basis = first_order_H.shape
    n_dim = n_atoms*n_cart

    kweights = kpoints[:,-1]

    G = np.zeros([n_dim,n_kpts,n_spin,n_states, n_basis],dtype=np.complex)
    counter = 0
    for a in range(n_atoms):
        for c in range(n_cart):
            for s in range(n_spin):
                G[counter,:,s,:,:] = first_order_H[a,c,:,s,:,:] - fermi_energy*first_order_S[a,c,:,:,:]
            counter += 1

    # calculate spectral components

    n_axis = int(keys['max_energy']/keys['discretization_length'])
    x_axis = np.array( [keys['discretization_length']*i for i in range(n_axis)] )
    ef = fermi_energy
    delta_method = keys['discretization_type']
    sigma = keys['discretization_broadening']
    T = keys['temperature']
    debug = keys['debug']

    spectral_function = np.zeros([n_dim*(n_dim+1)/2,n_axis],dtype=np.complex)

    counter = 0
    for d in range(n_dim):
        for d2 in range(d,n_dim):
            print('Calculating spectral function for components {0} and {1}'.format(d,d2))
            for s in range(n_spin):
                for k in range(n_kpts):
                    if debug:
                        print('s ', s, 'k ', k)
                    wk = kweights[k]
                    orb_min = 0
                    orb_lumo = 0
                    orb_homo = 0
                    orb_max = 0
                    for ei,e in enumerate(eigenvalues[k,s,:]):
                        occ = fermi_occ(e,ef,T)*(2./n_spin)
                        if e<=ef-2.00*keys['max_energy']:
                            orb_min = ei
                        if occ>=0.999:
                            orb_lumo = ei
                        if occ>= 0.001:
                            orb_homo = ei
                        if e<=ef+2.00*keys['max_energy']:
                            orb_max = ei
                    if debug:
                        print(orb_min, orb_homo, orb_lumo, orb_max)
                    for i in range(orb_min,orb_homo+1):
                        for f in range(orb_lumo, orb_max+1):
                            e = eigenvalues[k,s,f] - eigenvalues[k,s,i]
                            occ =(fermi_occ(eigenvalues[k,s,i],ef,T) - fermi_occ(eigenvalues[k,s,f],ef,T))*(2./n_spin)
                            if e>0.0 and e<=1.0*keys['max_energy'] and occ>=1.E-5:
                                #calculate transition strength
                                nacs1 = np.dot(psi[k,s,i,:].conjugate().transpose(), np.dot(G[d,k,s,:,:], psi[k,s,f,:]))
                                nacs2 = np.dot(psi[k,s,i,:].conjugate().transpose(), np.dot(G[d2,k,s,:,:], psi[k,s,f,:]))
                                nacs = np.dot(nacs1.conjugate().transpose(),nacs2)
                                nacs /= (e)
                                nacs *= wk
                                nacs *= occ
                                if debug:
                                    print(i, f, e, ' ' , occ, ' ', 
                                        (nacs*hbar*pi/(time_to_ps*sqrt(masses[d//3]*masses[d2//3]))).real, 
                                        ' ', (nacs*hbar*pi/(time_to_ps*sqrt(masses[d//3]*masses[d2//3]))).imag)
                                spectral_function[counter,:] += discretize_peak(e,nacs, x_axis, sigma, delta_method)
            
            spectral_function[counter,:] *= (pi*hbar)/sqrt(masses[d/3]*masses[d2/3])
            counter += 1

    return x_axis, spectral_function


def calculate_spectral_function_tensor_q(
        fermi_energy,
        eigenvalues,
        kpoints,
        psi,
        first_order_H,
        first_order_S,
        masses,
        atomic_positions,
        basis_pos,
        cell,
        **kwargs):
    """
    Calculates spectral functions for all cartesian directions and 
    couplings between directions. It does this for all possible combinations 
    of k points

    Parameters:

    fermi_energy: float
        Fermi Level in eV

    eigenvalues: np.array
        numpy array of eigenvalues with dimensions given by 
        n_kpoints, n_spin, n_states

    kpoints: np.array
        array of kpoints with shape (n_kpoints, 4) where the 
        first three columns give the x, y and z components of 
        the kvector in reciprocal space and the fourth column 
        gives the kpoint weight.

    psi: np.array
        array of wavefunction coefficients with its shape given 
        by n_kpoints, n_spin, n_states and n_basisfunctions

    first_order_H: np.array
        cartesian derivative of the hamiltonian matrix with its 
        6 dimension given by n_atoms, 3 cartesian directions, 
        n_kpoints, n_spin, n_basis, n_basis

    first_order_S: np.array
        cartesian derivative of the overlap matrix with its 
        6 dimension given by n_atoms, 3 cartesian directions, 
        n_kpoints, n_spin, n_basis, n_basis
        
    masses: np.array or list
        list of masses for the n_atoms active atoms

    atomic_positions: np.array
        list of atomic positions

    basis_pos: np.array
        list of positions for the basis functions

    cell : np.array
        cell vectors

    Returns:

    x_axis: np.array
        array with the discretized x axis values for the spectral function

    spectral_function:: np.array
        2-dimensional array 

    """

    #ANALYSE KWARGS

    default_keywords = {
            'discretization_type' : 'gaussian',
            'discretization_broadening' : 0.05,
            'discretization_length' : 0.01,
            'max_energy' : 3.0,
            'temperature': 300,
            'debug' : 1,
            }

    keys = {}
    for key in default_keywords.keys():
        if  key in kwargs:
            keys[key] = kwargs[key] 
        else:
            keys[key] = default_keywords[key]

    #build coupling matrix G
    
    n_atoms, n_cart, n_kpts, n_spin, n_states, n_basis = first_order_H.shape
    n_dim = n_atoms*n_cart

    kweights = kpoints[:,-1]
    kpts = kpoints[:,:3]

    #THIS IS WHERE CYTHON STUFF GOES
    print('Setting up the coupling matrix ')

    #prepare N vector
    rec_cell = 2*np.pi*np.linalg.inv(cell.T)
    rec_celli = np.linalg.inv(rec_cell)
    frac_kpts = np.dot(kpts,rec_celli)
    nmin = np.abs(frac_kpts[np.nonzero(frac_kpts)]).min()
    frac_kpts /= nmin
    nk = n_kpts
    ns = n_spin
    N = np.array(np.round(frac_kpts),dtype=np.float)
    for n, nvec in enumerate(N):
        N[n] = np.dot(nvec, cell)

    print('N')
    print(N)

    raise NotImplementedError('This is work in progress')

    # real_H_r = np.zeros([n_atoms,n_cart,nk,nk,ns,n_basis,n_basis],dtype=np.complex) 
    # real_S_r = np.zeros([n_atoms,n_cart,nk,nk,n_basis,n_basis],dtype=np.complex) 
    # for N1 in range(nk):
        # Nvec1 = N[N1] 
        # for N2 in range(nk):
            # Nvec2 = N[N2]
            # print 'working on N1 and N2  ', N1, N2
            # for l in range(nk):
                # kvec = kpts[l]
                # kw = kweights[l]
                # kx = kvec[0]*(Nvec2[0]-Nvec1[0])+ 
                     # kvec[1]*(Nvec2[1]-Nvec1[1])+ 
                     # kvec[2]*(Nvec2[2]-Nvec1[2])
                # phase = np.exp(1.0j*kx)             
                # real_H_r[:,:,N1,N2,:,:,:]+=first_order_H[:,:,l,:,:,:]*phase*kw
                # real_S_r[:,:,N1,N2,:,:]+=first_order_S[:,:,l,:,:]*phase*kw
    # print 'Built real space first_order_H and first_order_S'
    
    # G = np.zeros([n_atoms,n_cart,nk,nk,ns,n_basis,n_basis],dtype=np.complex)
    
    # counter = 0
    # for atom in range(n_atoms):
        # for cart in range(n_cart):
    # print 'calculating G for atom {0} and cart {1}'.format(atom,cart)
    # for s in range(ns):
                # for k1 in range(nk):
                    # print 'k1 ', k1
                    # kvec1 = kpts[k1]
                    # for k2 in range(nk):
                        # print 'k2 ', k2
                        # kvec2 = kpts[k2]
                        # for N1 in range(nk):
                            # Nvec1 = N[N1] 
                            # for N2 in range(nk):
                                # Nvec2 = N[N2]
                                # tmpH = np.zeros([n_atoms,n_cart,n_basis,n_basis],dtype=np.complex)
                                # tmpS = np.zeros([n_atoms,n_cart,n_basis,n_basis],dtype=np.complex)
                                # for l in range(nk):
                                    # kvec = kpts[l]
                                    # kw = kweights[l]
                                    # kx = kvec[0]*(Nvec2[0]-Nvec1[0])+ 
                                         # kvec[1]*(Nvec2[1]-Nvec1[1])+ 
                                         # kvec[2]*(Nvec2[2]-Nvec1[2])
                                    # phase = np.exp(1.0j*kx)             
                                    # tmpH += first_order_H[:,:,l,s,:,:]*phase*kw
                                    # tmpS += first_order_S[:,:,l,:,:]*phase*kw

                                # kx = kvec2[0]*Nvec2[0]-kvec1[0]*Nvec1[0]+ 
                                     # kvec2[1]*Nvec2[1]-kvec1[1]*Nvec1[1]+ 
                                     # kvec2[2]*Nvec2[2]-kvec1[2]*Nvec1[2]
                                # phase = np.exp(1.0j*kx)
                                # G[:,:,k1,k2,s,:,:] += (tmpH-fermi_energy*tmpS)*phase



    build_G.build_G(
            first_order_H,
            first_order_S,
            atomic_positions, 
            basis_pos,
            kpts, 
            kweights,
            N,
            fermi_energy,
            ) 

    print('Finished setting up the coupling matrix')

    assert 0

    # calculate spectral components

    n_axis = int(keys['max_energy']/keys['discretization_length'])
    x_axis = np.array( [keys['discretization_length']*i for i in range(n_axis)] )
    ef = fermi_energy
    delta_method = keys['discretization_type']
    sigma = keys['discretization_broadening']
    T = keys['temperature']
    debug = keys['debug']

    spectral_function = np.zeros([n_dim*(n_dim+1)/2,n_axis],dtype=np.complex)

    counter = 0
    for d in range(n_dim):
        for d2 in range(d,n_dim):
            if debug:
                print('Calculating spectral function for components {0} and {1}'.format(d,d2))
            for k1 in range(n_kpts):
                for k2 in range(n_kpts):
                    #TODO could be a good place to build G matrix for this set of k points

                    for s in range(n_spin):

                        if debug:
                            print('s ', s, 'k1 ', k1, 'k2 ', k2)
                        wk1 = kweights[k1]
                        wk2 = kweights[k2]
                        wk = wk1*wk2
                        orb_min = 0
                        orb_lumo = 0
                        orb_homo = 0
                        orb_max = 0
                        for ei,e in enumerate(eigenvalues[k1,s,:]):
                            occ = fermi_occ(e,ef,T)
                            if e<=ef-2.00*keys['max_energy']:
                                orb_min = ei
                            if occ>= 0.001:
                                orb_homo = ei
                        for ei,e in enumerate(eigenvalues[k2,s,:]):
                            occ = fermi_occ(e,ef,T)
                            if occ>=0.999:
                                orb_lumo = ei
                            if e<=ef+2.00*keys['max_energy']:
                                orb_max = ei
                        if debug:
                            print(orb_min, orb_homo, orb_lumo, orb_max)
                        for i in range(orb_min,orb_homo+1):
                            for f in range(orb_lumo, orb_max+1):
                                e = eigenvalues[k2,s,f] - eigenvalues[k1,s,i]
                                occ =fermi_occ(eigenvalues[k1,s,i],ef,T) - fermi_occ(eigenvalues[k2,s,f],ef,T)
                                if e>0.0 and e<=1.0*keys['max_energy'] and occ>=1.E-5:
                                    #calculate transition strength

                                    nacs1 = np.dot(psi[k1,s,i,:].conjugate().transpose(),np.dot(G[:,:],
                                            psi[k2,s,f,:]))
                                    nacs2 = np.dot(psi[k1,s,i,:].conjugate().transpose(),np.dot(G[:,:],
                                            psi[k2,s,f,:]))
                                    nacs = np.dot(nacs1.conjugate().transpose(),nacs2)
                                    nacs /= (e)
                                    nacs *= wk
                                    nacs *= occ*(3.-n_spin)
                                    if debug:
                                        print(i, f, e, ' ' , occ, ' ', 
                                            (nacs*hbar*pi/(time_to_ps*sqrt(masses[d//3]*masses[d2//3]))).real, 
                                            ' ', (nacs*hbar*pi/(time_to_ps*sqrt(masses[d//3]*masses[d2//3]))).imag)
                                    
                                    spectral_function[counter,:] += discretize_peak(e,nacs, x_axis, sigma, delta_method)
            
            spectral_function[counter,:] *= (pi*hbar)/sqrt(masses[d//3]*masses[d2//3])
            counter += 1


    return x_axis, spectral_function

def evaluate_friction_at_zero(
        fermi_energy,
        eigenvalues,
        kpoints,
        psi,
        first_order_H,
        first_order_S,
        masses,
        **kwargs):
    """
    Calculates spectral functions for all cartesian directions and 
    couplings between directions.

    Parameters:

    fermi_energy: float
        Fermi Level in eV

    eigenvalues: np.array
        numpy array of eigenvalues with dimensions given by 
        n_kpoints, n_spin, n_states

    kpoints: np.array
        array of kpoints with shape (n_kpoints, 4) where the 
        first three columns give the x, y and z components of 
        the kvector in reciprocal space and the fourth column 
        gives the kpoint weight.

    psi: np.array
        array of wavefunction coefficients with its shape given 
        by n_kpoints, n_spin, n_states and n_basisfunctions

    first_order_H: np.array
        cartesian derivative of the hamiltonian matrix with its 
        6 dimension given by n_atoms, 3 cartesian directions, 
        n_kpoints, n_spin, n_basis, n_basis

    first_order_S: np.array
        cartesian derivative of the overlap matrix with its 
        6 dimension given by n_atoms, 3 cartesian directions, 
        n_kpoints, n_spin, n_basis, n_basis
        
    masses: np.array or list
        list of masses for the n_atoms active atoms


    Returns:

    x_axis: np.array
        array with the discretized x axis values for the spectral function

    spectral_function:: np.array
        2-dimensional array 

    """

    #ANALYSE KWARGS

    default_keywords = {
            'delta_function_type' : 'gaussian',
            'delta_function_width': 0.60,
            'perturbing_energy' : 0.0,
            'max_energy' : 3.0,
            'temperature': 300,
            'debug' : 0,
            }

    keys = {}
    for key in default_keywords.keys():
        if  key in kwargs:
            keys[key] = kwargs[key] 
        else:
            keys[key] = default_keywords[key]

    #build coupling matrix G
    
    n_atoms, n_cart, n_kpts, n_spin, n_states, n_basis = first_order_H.shape
    n_dim = n_atoms*n_cart

    kweights = kpoints[:,-1]

    G = np.zeros([n_dim,n_kpts,n_spin,n_states, n_basis],dtype=np.complex)
    counter = 0
    for a in range(n_atoms):
        for c in range(n_cart):
            for s in range(n_spin):
                G[counter,:,s,:,:] = first_order_H[a,c,:,s,:,:] - fermi_energy*first_order_S[a,c,:,:,:]
            counter += 1

    # calculate spectral components
    ef = fermi_energy
    delta_method = keys['delta_function_type']
    sigma = keys['delta_function_width']
    perturbation_e = keys['perturbing_energy']
    T = keys['temperature']
    debug = keys['debug']

    friction_tensor = np.zeros([n_dim,n_dim],dtype=np.float)

    counter = 0
    for d in range(n_dim):
        for d2 in range(d,n_dim):
            friction = 0.0
            print('Calculating spectral function for components {0} and {1}'.format(d,d2))
            for s in range(n_spin):
                for k in range(n_kpts):
                    if debug:
                        print('s ', s, 'k ', k)
                    wk = kweights[k]
                    orb_min = 0
                    orb_lumo = 0
                    orb_homo = 0
                    orb_max = 0
                    for ei,e in enumerate(eigenvalues[k,s,:]):
                        occ = fermi_occ(e,ef,T)*(2./n_spin)
                        if e<=ef-2.00*keys['max_energy']:
                            orb_min = ei
                        if occ>=0.999:
                            orb_lumo = ei
                        if occ>= 0.001:
                            orb_homo = ei
                        if e<=ef+2.00*keys['max_energy']:
                            orb_max = ei
                    if debug:
                        print(orb_min, orb_homo, orb_lumo, orb_max)
                    for i in range(orb_min,orb_homo+1):
                        for f in range(orb_lumo, orb_max+1):
                            e = eigenvalues[k,s,f] - eigenvalues[k,s,i]
                            occ =(fermi_occ(eigenvalues[k,s,i],ef,T) - fermi_occ(eigenvalues[k,s,f],ef,T))*(2./n_spin)
                            if e>0.0 and e<=1.0*keys['max_energy'] and occ>=1.E-5:
                                #calculate transition strength
                                nacs1 = np.dot(psi[k,s,i,:].conjugate().transpose(),np.dot(G[d,k,s,:,:],
                                        psi[k,s,f,:]))
                                nacs2 = np.dot(psi[k,s,i,:].conjugate().transpose(),np.dot(G[d2,k,s,:,:],
                                        psi[k,s,f,:]))
                                nacs = np.dot(nacs1.conjugate().transpose(),nacs2)
                                nacs /= (e)
                                nacs *= wk
                                nacs *= occ
                                if debug:
                                    print(i, f, e, ' ' , occ, ' ',
                                        (nacs*hbar*pi/(time_to_ps*sqrt(masses[d/3]*masses[d2/3]))).real, 
                                        ' ', 
                                        (nacs*hbar*pi/(time_to_ps*sqrt(masses[d/3]*masses[d2/3]))).imag
                                        )
                                friction += nacs*delta_function(e, perturbation_e, sigma, delta_method)/(0.5*(1.-np.math.erf((-e/sigma)*(1./np.sqrt(2.)))))
                                #spectral_function[counter,:] += discretize_peak(e,nacs, x_axis, sigma, delta_method)
            friction_tensor[d,d2] = friction 
            friction_tensor[d,d2] *= (pi*hbar)/sqrt(masses[d/3]*masses[d2/3])
            friction_tensor[d2,d] = friction_tensor[d,d2]
            #spectral_function[counter,:] *= (pi*hbar)/sqrt(masses[d/3]*masses[d2/3])
            counter += 1

    return friction_tensor 


def evaluate_friction_at_zero_mode(fermi_energy, eigenvalues, kpoints, psi, first_order_H, first_order_S, **kwargs):
    """
    Calculates spectral functions for all cartesian directions and 
    couplings between directions.

    Parameters:

    fermi_energy: float
        Fermi Level in eV

    eigenvalues: np.array
        numpy array of eigenvalues with dimensions given by 
        n_kpoints, n_spin, n_states

    kpoints: np.array
        array of kpoints with shape (n_kpoints, 4) where the 
        first three columns give the x, y and z components of 
        the kvector in reciprocal space and the fourth column 
        gives the kpoint weight.

    psi: np.array
        array of wavefunction coefficients with its shape given 
        by n_kpoints, n_spin, n_states and n_basisfunctions

    first_order_H: np.array
        cartesian derivative of the hamiltonian matrix with its 
        6 dimension given by n_atoms, 3 cartesian directions, 
        n_kpoints, n_spin, n_basis, n_basis

    first_order_S: np.array
        cartesian derivative of the overlap matrix with its 
        6 dimension given by n_atoms, 3 cartesian directions, 
        n_kpoints, n_spin, n_basis, n_basis
        

    Returns:

    """

    #ANALYSE KWARGS

    default_keywords = {
            'delta_function_type' : 'gaussian',
            'delta_function_width': 0.60,
            'perturbing_energy' : 0.0,
            'max_energy' : 3.0,
            'temperature': 300,
            'debug' : 0,
            }

    keys = {}
    for key in default_keywords.keys():
        if  key in kwargs:
            keys[key] = kwargs[key] 
        else:
            keys[key] = default_keywords[key]

    #build coupling matrix G
    
    n_kpts, n_spin, n_states, n_basis = first_order_H.shape

    kweights = kpoints[:,-1]

    G = np.zeros([n_kpts,n_spin,n_states, n_basis],dtype=np.complex)
    for s in range(n_spin):
        G[:,s,:,:] = first_order_H[:,s,:,:] - fermi_energy*first_order_S[:,:,:]

    # calculate spectral components
    ef = fermi_energy
    delta_method = keys['delta_function_type']
    sigma = keys['delta_function_width']
    perturbation_e = keys['perturbing_energy']
    T = keys['temperature']
    debug = keys['debug']

    friction = 0.0
    for s in range(n_spin):
        for k in range(n_kpts):
            if debug:
                print('s ', s, 'k ', k)
            wk = kweights[k]
            orb_min = 0
            orb_lumo = 0
            orb_homo = 0
            orb_max = 0
            for ei,e in enumerate(eigenvalues[k,s,:]):
                occ = fermi_occ(e,ef,T)*(2./n_spin)
                if e<=ef-2.00*keys['max_energy']:
                    orb_min = ei
                if occ>=0.999:
                    orb_lumo = ei
                if occ>= 0.001:
                    orb_homo = ei
                if e<=ef+2.00*keys['max_energy']:
                    orb_max = ei
            if debug:
                print(orb_min, orb_homo, orb_lumo, orb_max)
            for i in range(orb_min,orb_homo+1):
                for f in range(orb_lumo, orb_max+1):
                    e = eigenvalues[k,s,f] - eigenvalues[k,s,i]
                    occ =(fermi_occ(eigenvalues[k,s,i],ef,T) - fermi_occ(eigenvalues[k,s,f],ef,T))*(2./n_spin)
                    if e>0.0 and e<=1.0*keys['max_energy'] and occ>=1.E-5:
                        #calculate transition strength
                        nacs1 = np.dot(psi[k,s,i,:].conjugate().transpose(),np.dot(G[k,s,:,:],
                                psi[k,s,f,:]))
                        nacs2 = np.dot(psi[k,s,i,:].conjugate().transpose(),np.dot(G[k,s,:,:],
                                psi[k,s,f,:]))
                        nacs = np.dot(nacs1.conjugate().transpose(),nacs2)
                        nacs /= (e)
                        nacs *= wk
                        nacs *= occ
                        if debug:
                            print(i, f, e, ' ' , occ, ' ',
                                (nacs*hbar*pi/(time_to_ps)).real, 
                                ' ', 
                                (nacs*hbar*pi/(time_to_ps)).imag
                                )
                        friction += nacs*delta_function(e, perturbation_e, sigma, delta_method)/(0.5*(1.-np.math.erf((-e/sigma)*(1./np.sqrt(2.)))))
                        #spectral_function[counter,:] += discretize_peak(e,nacs, x_axis, sigma, delta_method)
    
    friction *= (pi*hbar)

    return friction


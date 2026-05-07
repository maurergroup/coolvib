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
Parsing routines for FHI-Aims
"""

import numpy as np
import os
from os.path import join as pathjoin

def parse_aims_tensor(model, spin=True, path='./', filename='aims.out', active_atoms=[1], incr=0.01, debug=0):
    """
    This subroutine returns all necessary information stored in the aims output files
    and returns the fermi level, the eigenvalues, the kpoints and the hamiltonian and the 
    overlap matrices

    Parameters: 
    
        model : workflow_tensor class
            workflow_tensor class into which the data is supposed to be
            written

        spin : boolean
            True or False for spin collinear or no spin

        path : string
            path to the main directory that holds the data

        filename : string
            filename to the main output of the equilibrium run, 
            it is assumed that the equilibrium run lies at path/eq/filename

        active_atoms : list
            list of atoms which are to be included in the friction_tensor

        incr : float
            finite difference increment that was used to calculate the matrices

        debug : 0 / 1
            debug flag 0 = no, 1 = yes

    Output

        model: workflow_tensor class
            returns the workflow_tensor class including the read data

    """

    cell = model.atoms.get_cell()

    print('Reading eigenvalues and eigenvectors')
    fermi_level, kpoints_weights = aims_read_fermi_and_kpoints(pathjoin(path+'/eq', filename), cell)
    eigenvalues, psi, occ, basis_pos = \
            aims_read_eigenvalues_and_coefficients(fermi_level, directory=pathjoin(path,'eq'),\
            spin=spin, debug=debug)
    if spin:
        n_spin=2
    else:
        n_spin=1
    n_states = psi.shape[2]
    n_basis = psi.shape[3]

    H_q = np.zeros([len(active_atoms),3,len(kpoints_weights), n_spin, n_basis, n_basis],dtype=np.complex128)
    S_q = np.zeros([len(active_atoms),3,len(kpoints_weights), n_basis, n_basis],dtype=np.complex128)

    ai = 0
    for a in active_atoms:
        for c in range(3):
            print('Reading hamiltonian and overlap matrices for coordinate {0} {1}'.format(a, c))
            H_plus, S_plus = aims_read_HS(path+'/a{0}c{1}+/'.format(int(a),int(c)),spin=spin,debug=debug)
            H_minus, S_minus = aims_read_HS(path+'/a{0}c{1}-/'.format(int(a),int(c)),spin=spin,debug=debug)
            H_q[ai,c,:,:,:,:] = (H_plus - H_minus)/2.0/incr
            S_q[ai,c,:,:,:] = (S_plus - S_minus)/2.0/incr
        ai += 1


    model.eigenvalues = eigenvalues
    model.fermi_energy = fermi_level
    model.kpoints = kpoints_weights
    model.psi = psi
    model.basis_pos = basis_pos
    model.occ = occ
    model.first_order_H = H_q
    model.first_order_S = S_q

    return model

def parse_aims_mode(model, spin=True, path='./', prefix='mode', filename='aims.out', incr=0.01, debug=0):
    """
    This subroutine returns all necessary information stored in the aims output files
    and returns the fermi level, the eigenvalues, the kpoints and the hamiltonian and the 
    overlap matrices

    Parameters: 
    
        model : workflow_mode class
            workflow_mode class into which the data is supposed to be
            written

        spin : boolean
            True or False for spin collinear or no spin

        path : string
            path to the main directory that holds the data

        filename : string
            filename to the main output of the equilibrium run, 

        incr : float
            finite difference increment that was used to calculate the matrices

        debug : 0 / 1
            debug flag 0 = no, 1 = yes

    Output

        model: workflow_mode class
            returns the workflow_mode class including the read data
    """

    cell = model.atoms.get_cell()

    print('Reading eigenvalues and eigenvectors')
    fermi_level, kpoints_weights = aims_read_fermi_and_kpoints(pathjoin(path+'/'+prefix+'_eq', filename), cell)
    eigenvalues, psi, occ, basis_pos = \
            aims_read_eigenvalues_and_coefficients(fermi_level, directory=pathjoin(path,prefix+'_eq'),\
            spin=spin, debug=debug)
    if spin:
        n_spin=2
    else:
        n_spin=1
    n_states = psi.shape[2]
    n_basis = psi.shape[3]

    H_q = np.zeros([len(kpoints_weights), n_spin, n_basis, n_basis],dtype=np.complex128)
    S_q = np.zeros([len(kpoints_weights), n_basis, n_basis],dtype=np.complex128)


    print('Reading hamiltonian and overlap matrices')
    H_plus, S_plus = aims_read_HS(path+'/'+prefix+'_+',spin=spin,debug=debug)
    H_minus, S_minus = aims_read_HS(path+'/'+prefix+'_-',spin=spin,debug=debug)
    H_q[:,:,:,:] = (H_plus - H_minus)/2.0/incr
    S_q[:,:,:] = (S_plus - S_minus)/2.0/incr

    model.eigenvalues = eigenvalues
    model.fermi_energy = fermi_level
    model.kpoints = kpoints_weights
    model.psi = psi
    model.basis_pos = basis_pos
    model.occ = occ
    model.first_order_H = H_q
    model.first_order_S = S_q

    return model 


def aims_read_fermi_and_kpoints(filename,cell=None):
    """
    Reads the fermi_level and the kpoint information from the 
    main OUTPUT file of FHI-Aims. In order for this to work one needs 
    to set the keyword 'output k_point_list

    Parameters:

        filename: str
            FHI-Aims output file name

        cell: np.array
            array containing the unit cell data
   
    Output:

        fermi_level: float
            Fermi Energy of the system

        kpoint_weights: np.array
            Array of dimensions (n_kpoints, 4) that holds the 
            x, y and z components of the k vectors in reciprocal space and the 
            kpoint weights.

    """

    if cell is None:
        raise ValueError('Please supply unit cell for aims_read_fermi_kpoints')
    rcell = 2.*np.pi*np.linalg.inv(cell.T)

    with open(filename,'r') as f:
        
        kweights_exist = False
        fermi_level_exists = False

        content = f.readlines()
        for l, line in enumerate(content):
            if 'k_point_list' in line:
                print('Found k_point_list keyword, extracting kpoint_weights')
                kweights_exist = True

            if '| Chemical potential (Fermi level)' in line and not fermi_level_exists:
                print('Found Fermi Level, extracting fermi level')
                fermi_level_exists = True

        if fermi_level_exists and kweights_exist:

            kpoints = []
            fermi_level = 0.0
            k_points_are_done = False
            n_k = 0
            nkpts = -1
            for l, line in enumerate(content):
                if '| Number of k-points          ' in line:
                    nkpts = int(line.split()[-1])
                if '| k-point: ' in line and not k_points_are_done:
                    print(line)
                    kpoint = []
                    kpoint.append(float(line.split()[4]))
                    kpoint.append(float(line.split()[5]))
                    kpoint.append(float(line.split()[6]))
                    kpoint.append(float(line.split()[9]))
                    kpoints.append(kpoint)
                    n_k += 1
                if n_k == nkpts:
                    k_points_are_done = True
                if '| Chemical potential (Fermi level):' in line:
                    fermi_level = float(line.split()[-2])
            
            kpoint_weights = np.array(kpoints)
            #transform k points fromr reciprocal to absolute values
            for i in range(len(kpoint_weights)):
                kpoint_weights[i,:3] = np.dot(kpoint_weights[i,:3],rcell)
            return (fermi_level, kpoint_weights)

        else:
            raise RuntimeError('Cannot extract fermi level and/or kpoint_weights \n \
                    Either k_point_list keyword not set, or run is not converged \n \
                    or you are using an unsupported FHI-aims version.')

def aims_read_eigenvalues_and_coefficients(fermi_level, directory='./', spin=False, debug=False):
    """
    This routine reads eigenvalues and eigenvectors from aims output
    generated via the keywords 'output band x1 y1 z1 x2 y2 z2 2' and 
    'output eigenvectors'.

    arguments are the directory where the 
    KS_eigenvectors_XX.band_X.kpt_X.out files lie, the value of the fermi_level, 
    and a logical specifying if the calculation was spin-collinear or without spin.

    the routine returns the eigenvalues, eigenvectors, occupations and orbital positions
    in this order as np.arrays

    """

    #name_base = directory+'/KS_eigenvectors'
    name_base = 'KS_eigenvectors'
    #spin 0 is dn spin 1 is up
    spin_postfix = ['_dn', '_up']
    band_basis = 2

    eigenvector_files = []
    #how many files exist
    for file in os.listdir(directory):
        if file.startswith(name_base):
            eigenvector_files.append(directory+'/'+file)

    if spin:
        n_kpts = int(len(eigenvector_files)/2)
        n_spin = 2
    else:
        n_kpts = len(eigenvector_files)
        n_spin = 1

    name_base = directory + '/' + name_base

    #now we need to find out how many basis functions are used
    if spin:
        filename = name_base + '_dn.band_1.kpt_1.out'
    else:
        filename = name_base + '.band_1.kpt_1.out'

    first_basis_line = 7 #number of lines at the top of the file before the basis decomposition
    n_states_line = 3 #number of lines at the top of the file before the list of states
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        n_basis = len(lines) - first_basis_line
        n_states = len(lines[4].split()[n_states_line:])
    
    eigenvalues = np.zeros([n_kpts, n_spin, n_states])
    occ = np.zeros([n_kpts, n_spin, n_states])
    psi = np.zeros([n_kpts, n_spin, n_states, n_basis],dtype=complex)
    orbital_pos = np.zeros(n_basis,dtype=np.int32)
    
    #loop through all files
    for s in range(n_spin):
        prefix = name_base
        if spin:
            prefix += spin_postfix[s]
        
        #this now assumes 2 kpts per 'band'
        for k in range(n_kpts):
            band = int(k/band_basis)+1
            kp = k%band_basis+1
            filename = prefix + '.band_{0}.kpt_{1}.out'.format(band,kp)
            #opening file
            with open(filename,'r') as f:
                if debug:
                    print('Reading eigenvalues and psi from {0} '.format(filename))
                lines = f.readlines()
                
                #line 4 contains the eigenvalues
                eigenval_line = 4
                print(np.array(lines[eigenval_line].split()[n_states_line:]))
                eigenvalues[k,s,:] = np.array(lines[eigenval_line].split()[n_states_line:]).astype(np.float64)
                
                #line 5 contains the occupations
                occ_line = 5
                print(np.array(lines[occ_line].split()[n_states_line:]))
                occ[k,s,:] = np.array(lines[occ_line].split()[n_states_line:]).astype(np.float64)
                
                for i in range(n_basis):
                    read_psi = np.array(lines[first_basis_line+i].split()[6:]).astype(np.float64).reshape(-1,2)
                    psi[k,s,:,i] = read_psi[:,0] + 1j*read_psi[:,1]
                    orbital_pos[i] = int(lines[first_basis_line+i].split()[1]) -1
    
    return eigenvalues, psi, occ, orbital_pos

def aims_read_HS(directory='./', spin=False, debug=False):
    """
    This routine extracts hamiltonian and overlap matrix from FHI-Aims 
    outputfiles generated with the keyword 'output band ....' and 
    output 'output hamiltonian_matrix' and 'output overlap_matrix'

    the function takes as arguments the directory where these files can be found 
    and a logical specifying the spin-mode.

    the function returns the Hamiltonian and overlap matrix as np.arrays

    """

    name_base_H = 'KS_hamiltonian_matrix'
    name_base_S = 'KS_overlap_matrix'
    band_basis = 2

    H_files = []
    S_files = []
    #how many files exist
    for file in os.listdir(directory):
        if file.startswith(name_base_H):
            H_files.append(file)
        if file.startswith(name_base_S):
            S_files.append(file)

    assert len(H_files) == len(S_files)

    n_kpts = len(H_files)
    if spin:
        n_spin = 2
    else:
        n_spin = 1

    #now we need to find out how many basis functions are used
    filename = directory+'/'+name_base_S + '.band_1.kpt_1.out'
    with open(filename, 'r') as f:
        lines = f.readlines()
        n_basis = len(lines) - 2 
    
    H = np.zeros([n_kpts, n_spin, n_basis, n_basis],dtype=complex)
    S = np.zeros([n_kpts, n_basis, n_basis],dtype=complex)

    for k in range(n_kpts):
        band = int(k/band_basis)+1
        kp = k%band_basis+1
        filename_H = directory + '/'+name_base_H + '.band_{0}.kpt_{1}.out'.format(band,kp)
        filename_S = directory + '/'+name_base_S + '.band_{0}.kpt_{1}.out'.format(band,kp)
        
        #opening files
        with open(filename_S,'r') as f:
            if debug:
                print('Reading overlap_matrix from {0} '.format(filename_S))
            print(np.loadtxt(filename_S,skiprows=2))
            s = np.loadtxt(filename_S,skiprows=2).reshape([n_basis,n_basis,2])
            S[k, :, :] = s[:,:,0] + 1j*s[:,:,1]
        with open(filename_H,'r') as f:
            if debug:
                print('Reading hamiltonian matrix from {0} '.format(filename_H))
            h = np.loadtxt(filename_H,skiprows=2)
            if spin:
                h_dn = h[:n_basis].reshape([n_basis,n_basis,2])
                h_up = h[n_basis:].reshape([n_basis,n_basis,2])
                H[k, 0, :, :] = h_dn[:,:,0] + 1j*h_dn[:,:,1]
                H[k, 1, :, :] = h_up[:,:,0] + 1j*h_up[:,:,1]
            else:
                h = h.reshape([n_basis,n_basis,2])
                H[k, 0, :, :] = h[:,:,0] + 1j*h[:,:,1]

    return H*27.211384500, S


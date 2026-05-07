cimport cython
import numpy as np
cimport numpy as np
REAL_TYPE = np.float
INT_TYPE = np.int
COMPLEX_TYPE = np.complex
ctypedef np.int_t INT_TYPE_t
ctypedef np.float_t REAL_TYPE_t
ctypedef np.complex_t COMPLEX_TYPE_t
from cmath import exp, pi
from libc.math cimport sqrt, cos, sin
from coolvib.constants import hbar_1, sqrt_pi, conversion_factor

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def build_realspace_HS(
        np.ndarray[REAL_TYPE_t, ndim=4] Hr, 
        np.ndarray[REAL_TYPE_t, ndim=4] Hi, 
        np.ndarray[REAL_TYPE_t, ndim=3] Sr, 
        np.ndarray[REAL_TYPE_t, ndim=3] Si, 
        np.ndarray[REAL_TYPE_t, ndim=2] atomic_positions, 
        np.ndarray[INT_TYPE_t,  ndim=1] basis_pos, 
        np.ndarray[REAL_TYPE_t, ndim=2] kpts, 
        np.ndarray[REAL_TYPE_t, ndim=1] kweights,
        np.ndarray[REAL_TYPE_t, ndim=2] N,
        ):
    """
    This function builds the real space version of the finite difference 
    matrix Gq
    """
    
    cdef Py_ssize_t a, b, l, k, N1, N2, s  
    cdef INT_TYPE_t n_basis, nk, ns
    cdef REAL_TYPE_t kw,co,si
    cdef REAL_TYPE_t kx, kx2
    cdef REAL_TYPE_t Htmp_r, Htmp_i, Stmp_r, Stmp_i
    cdef np.ndarray[REAL_TYPE_t,ndim=1] kvec = np.empty(3,dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=1] Nvec1 = np.empty(3,dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=1] Nvec2 = np.empty(3,dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=1] RNvec1 = np.empty(3,dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=1] RNvec2 = np.empty(3,dtype=REAL_TYPE)
    
    ns = Hr.shape[1]
    n_basis = len(basis_pos) 
    nk = len(kpts)
    
    #cdef np.ndarray[REAL_TYPE_t,ndim=4] H_r = np.zeros([ns,nk,n_basis,n_basis],dtype=REAL_TYPE)
    #cdef np.ndarray[REAL_TYPE_t,ndim=4] H_i = np.zeros([ns,nk,n_basis,n_basis],dtype=REAL_TYPE)
    #cdef np.ndarray[REAL_TYPE_t,ndim=3] S_r = np.zeros([nk,n_basis,n_basis],dtype=REAL_TYPE)
    #cdef np.ndarray[REAL_TYPE_t,ndim=3] S_i = np.zeros([nk,n_basis,n_basis],dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=5] H_r = np.zeros([ns,nk,nk,n_basis,n_basis],dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=5] H_i = np.zeros([ns,nk,nk,n_basis,n_basis],dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=4] S_r = np.zeros([nk,nk,n_basis,n_basis],dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=4] S_i = np.zeros([nk,nk,n_basis,n_basis],dtype=REAL_TYPE)

    #MOST INNER PART, H and S
    for s in range(ns):
        for N1 in range(nk):
            Nvec1 = N[N1] 
            for N2 in range(nk):
                Nvec2 = N[N2]
                for a in range(n_basis):
                    for b in range(n_basis):
                        RNvec1[:] = atomic_positions[basis_pos[a]]+Nvec1[:]
                        RNvec2[:] = atomic_positions[basis_pos[b]]+Nvec2[:]
                        Htmp_r = 0.0
                        Stmp_r = 0.0
                        Htmp_i = 0.0
                        Stmp_i = 0.0
                        #Fourier transformation of G matrix element
                        for l in range(nk):
                            kvec = kpts[l]
                            kw = kweights[l]
                            kx = kvec[0]*(RNvec2[0]-RNvec1[0])+ \
                                 kvec[1]*(RNvec2[1]-RNvec1[1])+ \
                                 kvec[2]*(RNvec2[2]-RNvec1[2])
                            co = cos(-kx)
                            si = sin(-kx)
                            Htmp_r += (co*Hr[l,s,a,b]-si*Hi[l,s,a,b]) *kw
                            Htmp_i += (co*Hi[l,s,a,b]+si*Hr[l,s,a,b]) *kw
                            Stmp_r += (co*Sr[l,a,b]-si*Si[l,a,b]) *kw
                            Stmp_i += (co*Si[l,a,b]+si*Sr[l,a,b]) *kw
                        H_r[s,N1,N2,a,b] = Htmp_r
                        H_i[s,N1,N2,a,b] = Htmp_i
                        S_r[N1,N2,a,b]   = Stmp_r
                        S_i[N1,N2,a,b]   = Stmp_i 
                        #H_r[s,N2,a,b] = Htmp_r
                        #H_i[s,N2,a,b] = Htmp_i
                        #S_r[N2,a,b]   = Stmp_r
                        #S_i[N2,a,b]   = Stmp_i 

    return H_r, H_i, S_r, S_i


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def build_realspace_Gq(
        np.ndarray[COMPLEX_TYPE_t, ndim=4] H_plus, 
        np.ndarray[COMPLEX_TYPE_t, ndim=4] H_minus, 
        np.ndarray[COMPLEX_TYPE_t, ndim=3] S_plus, 
        np.ndarray[COMPLEX_TYPE_t, ndim=3] S_minus, 
        np.ndarray[REAL_TYPE_t, ndim=2] atomic_positions, 
        np.ndarray[INT_TYPE_t,  ndim=1] basis_pos, 
        np.ndarray[REAL_TYPE_t, ndim=2] kpts, 
        np.ndarray[REAL_TYPE_t, ndim=1] kweights,
        np.ndarray[REAL_TYPE_t, ndim=2] N,
        np.ndarray[REAL_TYPE_t, ndim=2] Q,
        float dq, 
        float fermi_level, 
        ):
    """
    This function builds the real space version of the finite difference 
    matrix Gq
    """
    
    cdef Py_ssize_t a, b, l, k, N1, N2, s  
    cdef INT_TYPE_t n_basis, nk, ns
    cdef REAL_TYPE_t kw, co, si, tmp_r, tmp_i
    cdef REAL_TYPE_t kx, kx2, G_q_r,G_q_i
    cdef np.ndarray[REAL_TYPE_t,ndim=1] kvec = np.empty(3,dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=1] qvec1 = np.empty(3,dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=1] qvec2 = np.empty(3,dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=1] Nvec1 = np.empty(3,dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=1] Nvec2 = np.empty(3,dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=1] RNvec1 = np.empty(3,dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=1] RNvec2 = np.empty(3,dtype=REAL_TYPE)
    
    ns = H_plus.shape[1]
    n_basis = len(basis_pos) 
    nk = len(kpts)
    
    cdef np.ndarray[REAL_TYPE_t,ndim=5] Gq_r = np.empty([ns,nk,nk,n_basis,n_basis],dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=5] Gq_i = np.empty([ns,nk,nk,n_basis,n_basis],dtype=REAL_TYPE)
    
    #MOST INNER PART, H and S
    for s in range(ns):
        for N1 in range(len(N)):
            Nvec1 = N[N1]
            for N2 in range(len(N)):
                Nvec2 = N[N2]
                for a in range(n_basis):
                    for b in range(n_basis):
                        RNvec1[:] = atomic_positions[basis_pos[a]]+Nvec1[:]
                        RNvec2[:] = atomic_positions[basis_pos[b]]+Nvec2[:]
                        qvec1 = Q[basis_pos[a]]
                        qvec2 = Q[basis_pos[b]]
                        G_q_r = 0.0; G_q_i = 0.0;
                        #Fourier transformation of G matrix element
                        for l in range(nk):
                            kvec = kpts[l]
                            kw = kweights[l]
                            kx = kvec[0]*(RNvec1[0]+qvec1[0]-RNvec2[0]-qvec2[0])+ \
                                  kvec[1]*(RNvec1[1]+qvec1[1]-RNvec2[1]-qvec2[1])+ \
                                  kvec[2]*(RNvec1[2]+qvec1[2]-RNvec2[2]-qvec2[2])
                            #kx = kvec[0]*(RNvec1[0]-RNvec2[0])+ \
                                 #kvec[1]*(RNvec1[1]-RNvec2[1])+ \
                                 #kvec[2]*(RNvec1[2]-RNvec2[2])
                            co = cos(kx)
                            si = sin(kx)
                            tmp_r = (H_plus[l,s,a,b].real-fermi_level*S_plus[l,a,b].real)
                            tmp_i = (H_plus[l,s,a,b].imag-fermi_level*S_plus[l,a,b].imag)
                            kx = kvec[0]*(RNvec1[0]-qvec1[0]-RNvec2[0]+qvec2[0])+ \
                                  kvec[1]*(RNvec1[1]-qvec1[1]-RNvec2[1]+qvec2[1])+ \
                                  kvec[2]*(RNvec1[2]-qvec1[2]-RNvec2[2]+qvec2[2])
                            co = cos(kx)
                            si = sin(kx)
                            tmp_r -= (H_minus[l,s,a,b].real-fermi_level*S_minus[l,a,b].real)
                            tmp_i -= (H_minus[l,s,a,b].imag-fermi_level*S_minus[l,a,b].imag)
                            G_q_r += (co*tmp_r-si*tmp_i)*kw
                            G_q_i += (co*tmp_i+si*tmp_r)*kw
                        Gq_r[s,N1,N2,a,b] = G_q_r /(2*dq)
                        Gq_i[s,N1,N2,a,b] = G_q_i /(2*dq)

    return Gq_r, Gq_i

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def calculate_nonadiabatic_coupling_HS_kpts(
        np.ndarray[COMPLEX_TYPE_t, ndim=4] psi, 
        np.ndarray[REAL_TYPE_t, ndim=3] eigenvalues, 
        np.ndarray[REAL_TYPE_t, ndim=5] Gq_r, 
        np.ndarray[REAL_TYPE_t, ndim=5] Gq_i, 
        np.ndarray[REAL_TYPE_t, ndim=2] atomic_positions, 
        np.ndarray[INT_TYPE_t,  ndim=1] basis_pos, 
        np.ndarray[REAL_TYPE_t, ndim=2] N,
        np.ndarray[REAL_TYPE_t, ndim=2] kpts,
        np.ndarray[REAL_TYPE_t, ndim=1] kweights,
        float fermi_level, float delta, 
        inside_window, dirac_weight
        ):
    """
    Cython extension to calculate a general nonadiabatic coupling element in the 
    H-eS formalism of Head-Gordon where states can be at differen k points.
    """
  
    cdef Py_ssize_t a, b, l, k, N1, N2, k1, k2, s, i , f
    cdef Py_ssize_t orb_min_k1,  orb_min_k2, orb_max_k1, orb_max_k2
    cdef INT_TYPE_t n_basis, nk, ns
    cdef REAL_TYPE_t product_real,product_r, product_i
    cdef REAL_TYPE_t tmp_r, tmp_i, tmp2_r, tmp2_i
    cdef REAL_TYPE_t kx, wk1, wk2, e, co, si
    cdef REAL_TYPE_t psi_1_r, psi_1_i, psi_2_r, psi_2_i
    cdef np.ndarray[REAL_TYPE_t,ndim=1] Kvec1 = np.empty(3,dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=1] Kvec2 = np.empty(3,dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=1] Nvec1 = np.empty(3,dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=1] Nvec2 = np.empty(3,dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=1] RNvec1 = np.empty(3,dtype=REAL_TYPE)
    cdef np.ndarray[REAL_TYPE_t,ndim=1] RNvec2 = np.empty(3,dtype=REAL_TYPE)

    ns = eigenvalues.shape[1]
    nbasis = len(basis_pos) 
    product_r = 0.0
    product_i = 0.0
    nk = len(N)
    #first loop over states from image cells from which to excite
    gamma=0
    hit_the_window=0
    for s in range(ns):
        for k1, Kvec1 in enumerate(kpts):
            wk1 = kweights[k1]
            eigenvalues_k1 = eigenvalues[k1,s,:]
            #identify loop boundaries
            for ei,e in enumerate(eigenvalues_k1):
                if e<fermi_level-5.0:
                    orb_min_k1 = ei
                if e<fermi_level:
                    orb_max_k1 = ei+1
            #second loop over states from image cells into which to excite
            for k2, Kvec2 in enumerate(kpts):
                wk2 = kweights[k2]
                eigenvalues_k2 = eigenvalues[k2,s,:]
                #identify loop boundaries
                for ei,e in enumerate(eigenvalues_k2):
                    if e>fermi_level:
                        orb_min_k2 = ei
                        break
                for ei,e in enumerate(eigenvalues_k2):
                    if e>fermi_level+5.0:
                        orb_max_k2 = ei
                        break
                product = 0.0
                #first loop over states from which to excite
                for i in reversed(range(orb_min_k1,orb_max_k1)):
                    #second loop over states into which to excite
                    for f in range(orb_min_k2,orb_max_k2):
                        e = eigenvalues[k2,s,f] - eigenvalues[k1,s,i]
                        if inside_window(e):
                            product_r = 0.
                            product_i = 0.
                            hit_the_window += 1
                            for a in range(nbasis):
                                for b in range(nbasis):
                                    psi_1_r = psi[k1,s,i,a].real
                                    psi_1_i = psi[k1,s,i,a].imag
                                    psi_2_r = psi[k2,s,f,b].real
                                    psi_2_i = psi[k2,s,f,b].imag
                                    tmp_r = 0.0;tmp_i = 0.0;tmp2_r = 0.0;tmp2_i = 0.0;
                                    #MOST INNER PART, H and S
                                    for N1 in range(nk):
                                        Nvec1 = N[N1]
                                        for N2 in range(nk):
                                            Nvec2 = N[N2]
                                            RNvec1[:] = atomic_positions[basis_pos[a]]+Nvec1[:]
                                            RNvec2[:] = atomic_positions[basis_pos[b]]+Nvec2[:]
                                            kx = Kvec2[0]*Nvec2[0]+ \
                                                 Kvec2[1]*Nvec2[1]+ \
                                                 Kvec2[2]*Nvec2[2]
                                            kx -= Kvec1[0]*Nvec1[0]+ \
                                                  Kvec1[1]*Nvec1[1]+ \
                                                  Kvec1[2]*Nvec1[2]
                                            co = cos(kx)
                                            si = sin(kx)
                                            #Fourier transformation of G matrix element
                                            tmp_r += co*Gq_r[s,N1,N2,a,b]-si*Gq_i[s,N1,N2,a,b]
                                            tmp_i += co*Gq_i[s,N1,N2,a,b]+si*Gq_r[s,N1,N2,a,b]
                                    #multiply with wavefunc
                                    tmp2_r = tmp_r*psi_2_r-tmp_i*psi_2_i
                                    tmp2_i = tmp_r*psi_2_i+tmp_i*psi_2_r
                                    tmp_r = psi_1_r*tmp2_r-psi_1_i*tmp2_i
                                    tmp_i = psi_1_r*tmp2_i+psi_1_i*tmp2_r
                                    product_r += tmp_r
                                    product_i += tmp_i
                            product_real = product_r*product_r + product_i*product_i
                            product_real /= (e)
                            print(k1,k2, i, f, e, "{0:18.16f}".format(product_real*wk1*wk2))
                            product_real *= dirac_weight(e)
                            product += product_real
                gamma += product*wk1*wk2
    return  gamma, hit_the_window







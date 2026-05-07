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

#!/usr/bin/python
"""
integrate_tensor.py takes a given spectral function file 
calculated for all cartesian directions and calculates the 
friction tensor and analyzes the window convergence

python integrate_tensor.py <filename>

Arguments:

    <filename>: str
        Name of the file containing the spectral functions

Code Example::

    python integrate_tensor.py spectral_function.out

"""
def sin2(e,t):
    if e<0.0001 or t<0.000001:
        return 1.0
    else:
        hbar = 0.50 #0.6582/1000.0 #eV*ps
        x = (e*t)/(2*hbar)
        return (np.sin(x)**2)/(x*x)


from sys import argv

import numpy as np
from math import factorial

from coolvib.routines import *
if __name__=="__main__":


    f = open(str(argv[1]))#'nacs-spectrum.out')
    mode = argv[2]

    n = int(f.readline().split()[-1])
    dx = float(f.readline().split()[-1])
    nbins = int(f.readline().split()[-1])

    xaxis = np.zeros(nbins)
    for i,xx in enumerate(xaxis):
        xaxis[i] = i*dx
    nspectra = (n+1)*n/2

    spectrum = np.zeros([nspectra,nbins],dtype=np.complex)

    for i in range(nspectra):
        f.readline()
        f.readline()
        f.readline()
        for b in range(nbins):
            tmp = f.readline().split()
            if len(tmp)>2:
                spectrum[i,b] = float(tmp[-2])+1.0j*float(tmp[-1])
            else:
                spectrum[i,b] = float(tmp[-1])


    friction_tensor = np.zeros([n,n,5],dtype=np.complex)

    x0 = 0.0

    #window
    ss = [i*0.01+0.01 for i in range(60) ]

    names = ['square','gauss','fermi','sine','lorentz']
    print('window   square     gauss     fermi     sine     lorentz')
    for s in ss:
        k = 0
        for i in range(n):
            for j in range(i,n):
                norm = 0.
                friction = 0.+1.0j*0.
                for b in range(nbins):
                    delta = square(xaxis[b],x0,s) 
                    norm += delta
                    friction += delta*xaxis[b]*spectrum[k,b]
                friction /= norm
                friction_tensor[i,j,0] = friction
                friction_tensor[j,i,0] = np.conjugate(friction)
                norm = 0.
                friction = 0.+1.0j*0.
                for b in range(nbins):
                    delta = gaussian(xaxis[b],x0,s) 
                    norm += delta
                    #friction += delta*xaxis[b]*spectrum[k,b]
                    friction += delta*spectrum[k,b]
                friction /= norm
                friction_tensor[i,j,1] = friction
                friction_tensor[j,i,1] = np.conjugate(friction)
                norm = 0.
                friction = 0.+1.0j*0.
                for b in range(nbins):
                    delta = squashed_fermi(xaxis[b],x0,s) 
                    norm += delta
                    #friction += delta*xaxis[b]*spectrum[k,b]
                    friction += delta*spectrum[k,b]
                friction /= norm
                friction_tensor[i,j,2] = friction
                friction_tensor[j,i,2] = np.conjugate(friction)
                norm = 0.
                friction = 0.+1.0j*0.
                for b in range(nbins):
                    delta = sine(xaxis[b],x0,s) 
                    norm += delta
                    #friction += delta*xaxis[b]*spectrum[k,b]
                    friction += delta*spectrum[k,b]
                friction /= norm
                friction_tensor[i,j,3] = friction
                friction_tensor[j,i,3] = np.conjugate(friction)
                norm = 0.
                friction = 0.+1.0j*0.
                for b in range(nbins):
                    delta = lorentzian(xaxis[b],x0,s) 
                    norm += delta
                    #friction += delta*xaxis[b]*spectrum[k,b]
                    friction += delta*spectrum[k,b]
                friction /= norm
                friction_tensor[i,j,4] = friction
                friction_tensor[j,i,4] = np.conjugate(friction)
                k = k + 1

        #throw away imaginary parts
        friction_tensor = np.array(friction_tensor,dtype=np.float)
        #string = str(s) + ' '
        #for i in friction_tensor[0,0]:
            #string += str(1./i) + ' '
        #print string 
        #continue

        type = {'square':0, 'gaussian':1, 'fermi':2, 'sine':3,'lorentzian':4}

        string = str(s) + ' '
        for e in range(n):
            #for f in range(5):
            E,V = np.linalg.eig(friction_tensor[:,:,type[mode]])
            E = 1./E
            sort = np.argsort(E)
            E = np.array(E[sort],dtype=np.float)
            string += str(E[e]) + ' '
        
        mode1 = np.array([0., 0., 1.0, 0., 0., 1.0]) 
        mode1 /= np.linalg.norm(mode1)
        mode2 = np.array([0.,0.,1.0,0.,0.,-1.0])
        mode2 /= np.linalg.norm(mode2)
       
        string += str(1./(np.dot(mode1,np.dot(friction_tensor[:,:,type[mode]],mode1)))) + ' '
        string += str(1./(np.dot(mode2,np.dot(friction_tensor[:,:,type[mode]],mode2)))) + ' '

        print(string)

    
else:
    pass

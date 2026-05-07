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
This script takes an FHI-AIMS output generated with 
keyword 'output k_point_list' and prints the necessary 
input parameters for the control.in file that enable writing 
of Hamiltonian, Overlap and eigenvectors.

Execute this script the following way::

    fhiaims_add_outputoptions_to_control.py <OUTPUT_filename>

OUTPUT_filename has to be an FHI-AIMS OUTPUT file generated with control.in keyword
'output k_point_list'
"""
#!/usr/bin/python

from sys import argv, exit
import numpy as np

if __name__=="__main__":

    if len(argv)<2:
        print('Execute this script the following way:')
        print('fhiaims_add_outputoptions_to_control.py <OUTPUT_filename>')
        print('OUTPUT_filename has to be an FHI-AIMS OUTPUT file generated with keyword ')
        print("'output k_point_list '")
    else:
        print('Reading FHI-AIMS Output file')

        filename = argv[1]
        lines = open(filename,'r').readlines()
        kpoints = []
        for l,line in enumerate(lines):
            #if 'k_point_list' in line:
            #    pass
            #else:
            #    raise RuntimeError('OUTPUT file has to have been generated with \
            #keyword: output k_point_list.')

            if '| Number of k-points           ' in line:
                nkpts = int(line.split()[-1])
                n_line = l+1
            #if 'K-points in task  ' in lines[n_line]:
                    #nkpts += int(lines[n_line].split()[-1])
                    #n_line += 1
            if '| k-point: ' in line:
                kpoint = []
                kpoint.append(float(line.split()[4]))
                kpoint.append(float(line.split()[5]))
                kpoint.append(float(line.split()[6]))
                kpoint.append(float(line.split()[9]))
                kpoints.append(kpoint)

        kpoints = np.array(kpoints)

        print(kpoints)
        print('')
        n_kpts = len(kpoints)
        n_bands = int(n_kpts/2)
        
        output = '#add this to your control.in file'
        output += 'output k_point_list \n'
        
        for n in range(n_bands):
            output += 'output band  '
            for i in range(2):
                for xyz in range(3):
                    output += ' {0:8.4f} '.format(kpoints[n*2+i,xyz])
            output += ' 2 \n'

        output += 'output eigenvectors \n ' 
        output += 'output overlap_matrix \n ' 
        output += 'output hamiltonian_matrix \n ' 

        print(output)

        print(' ')
        output = '#add this to your python finite difference script using the ASE FHI-aims calculator\n'
        output += 'output = ["eigenvectors", '
        output += ' "k_point_list", '
        output += ' "hamiltonian_matrix", '
        output += ' "overlap_matrix", \n'
        for n in range(n_bands):
            output += ' "band  '
            for i in range(2):
                for xyz in range(3):
                    output += ' {0:8.4f} '.format(kpoints[n*2+i,xyz])
            output += ' 2 ", \n'
        output += ' ], '
        print(output)

else:
    pass

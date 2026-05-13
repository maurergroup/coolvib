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
parser module

This module contains all necessary routines to read
precalculated data from quantum chemistry packages. 

There are routines that bundle the IO for certain purposes, such as 
:py:func:`coolvib.parser.aims.parse_aims_tensor`, but one can also 
use the individual routines.

"""

codes = {
        'aims': 'parse_aims',
        'siesta': 'parse_siesta',
        } 

code_type = {
        'aims' : 'local',
        'siesta' : 'local',
        }

from coolvib.parser.siesta import parse_siesta_tensor, parse_siesta_mode
from coolvib.parser.aims import parse_aims_tensor, parse_aims_mode 
from coolvib.parser.aims import aims_read_elsi_to_csc, aims_read_elsi_density_matrix
from coolvib.parser.aims import find_num_atoms_and_kpoints

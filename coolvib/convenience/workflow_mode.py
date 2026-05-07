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
workflow_mode.py

Defines a class **workflow_mode** that acts as a data container and workflow 
guideline to calculate the non-adiabatic friction induced lifetime 
of a given vibrational mode.
"""
import coolvib
from coolvib.constants import time_to_ps
import coolvib.parser as parser
import coolvib.routines.friction_tensor as tensor
import coolvib.routines.spectral_function as spectral
import coolvib.routines.output as output
from ase.atoms import Atoms
import numpy as np

class workflow_mode():
    def __init__(self, atoms, code='aims', mode=None):
        """
        workflow_mode object.
        
        The workflow_mode objet is a basic data container and workflow guideline 
        for all calculations necessary to generate a friction tensor and 
        calculate vibrational lifetimes due to non-adiabatic couplings. 

        Parameters:

        atoms: ase.atoms class 
            defines the atoms, molecules and or surfaces for 
            which the calculation is performed.

        code: str 
            A string that defines from which quantum chemistry package the input 
            is received.

        mode: list
            A list or np.array of displacements that defines the mode along which the 
            lifetime is to be calculated.
           
        Examples:

            This is how this class is initialized::

                from ase.all import Atoms
                a = Atoms("CO", [(0.,0.,0.), (0.,0.,1.0)])
                model = workflow_mode(a, code='siesta', mode=[0.,0.,1.0, 0.,0.,-1.0])

        """
        if isinstance(atoms, Atoms):
            self.atoms = atoms
        else:
            raise ValueError('model needs an instance of an ase Atoms class')
        self.code = code
        if mode is not None:
            self.mode = np.array(mode)
            self.mode = self.mode/np.linalg.norm(self.mode)

        self.input_read=False

    def set_code(self, code):
        """
        Sets the code keyword
        """

        if code in codes:
            self.code = code
        else:
            print('This code is not supported, choose one of ', codes)

    def set_mode(self, mode=None):
        """
        Sets the active_atoms list
        """

        self.mode = mode

    def read_input_data(self, **kwargs):
        """
        Reads all necessary input data including first order matrices
        wavefunctions, eigenvalues, fermi_level, etc.

        This is a wrapper function for specific parser routines, depending 
        on which code is used. For detailed input explanations please see 
        the documentation in the :py:coolvib.parser: package.

        """

        if self.code is None:
            print('Please set code first using .set_code')

        method_to_call =getattr(parser,parser.codes[self.code]+'_mode')
        self = method_to_call(self, **kwargs)

        self.input_read = True
    
    def calculate_friction_from_spectrum(self,mode='default',**kwargs):
        """
        Calculates the friction for a given window and delta function, 
        assumes that the spectral function has already been calculated, otherwise 
        calculates it.

        Parameters:

        mode: str

        This is a wrapper function for :py:coolvib.routines.friction_tensor.calculate_friction:
        For detailed documentation please look there.
        """

        calculate_spec = True
        if hasattr(self, 'spectral_function'):
            calculate_spec = False
        
        #first calculate spectral function
        if calculate_spec:
            self.calculate_spectral_function(mode,**kwargs)

        ###calculate the tensor by applying the delta function
        x_axis = self.x_axis
        spectral_function = self.spectral_function

        self.friction_tensor = tensor.calculate_tensor(
                1,
                x_axis,
                spectral_function,
                **kwargs)

    def calculate_spectral_function(self,mode='default', **kwargs):
        """
        initiates calculation of the spectral functions and 
        takes the same arguments as coolvib.routines.calculate_spectral_function
     
        Parameters:

        mode : str
            'default' or 'momentum' for q=0 or q not 0 versions
        
        """

        #masses = self.atoms.get_masses()[self.active_atoms]
        #CASE CODE
        if parser.code_type[self.code] is 'local':
            #CASE mode
            if mode is 'default':
                self.x_axis, self.spectral_function = spectral.calculate_spectral_function_mode(
                    self.fermi_energy,
                    self.eigenvalues,
                    self.kpoints,
                    self.psi,
                    self.first_order_H,
                    self.first_order_S,
                    **kwargs)
            elif mode is 'momentum':
                pass
            #self.x_axis, self.spectral_function = spectral.calculate_spectral_function_mode_q(
            #        self.fermi_energy,
            #        self.eigenvalues,
            #        self.kpoints,
            #        self.psi,
            #        self.first_order_H,
            #        self.first_order_S,
            #        masses,
            #        self.basis_pos,
            #        **kwargs)

        else:
            #plane wave stuff
            raise NotImplementedError('calculate_spectral_function: code is not supported!')


    def print_spectral_function(self, filename='nacs-spectrum.out'):
        """
        prints spectral function to a file

        Parameters:

        filename: str
            name of the output file, default is nacs-spectrum.out
        """

        output.print_spectral_function(
                self.x_axis, self.spectral_function,1,
                filename)
    

    def calculate_friction(self, **kwargs):
        """

        """
        self.friction_tensor = spectral.evaluate_friction_at_zero_mode(
            self.fermi_energy,
            self.eigenvalues,
            self.kpoints,
            self.psi,
            self.first_order_H,
            self.first_order_S,
            **kwargs)


    def analyse_friction(self):
        """
        diagonalizes friction tensor and does spectral analysis
        """

        if hasattr(self, 'friction_tensor'):
            pass
        else:
            self.calculate_friction_tensor()

        print('Friction') 
        print(self.friction_tensor/time_to_ps)
        print('Lifetime')
        print(1./(self.friction_tensor/time_to_ps))

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 11:49:28 2023

@author: Korisnik
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:30:18 2023

@author: Korisnik
"""
import numpy as np
from params import PRICES, DISCOUNT_RATE, DISSATISFACTION_COEFFICIENTS
from math import comb

class ValueFunctions:
    def __init__(self):
        self.V = {}
                    
    def true_test(self, x):
        """
        Calculating feature transformation as the true reward for testing
        purposes

        Parameters
        ----------
        x : numpy array
            State.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        nv = []
        for el in x:
            bee = -el[2]*el[0]+ el[2]*el[3]-el[2]*el[1]
            disc = 0
            for dev in range(0, len(el[4:])):
                disc -= (el[4 + dev]**2)*DISSATISFACTION_COEFFICIENTS[dev]
            new = max(0, bee) + disc
            nv.append(new)
        return np.array(nv).reshape((-1,1))
    
    def basis_1(self, x):
        """
        Calculating feature transformation as the difference between base and 
        consumption and squared dissatisfaction

        Parameters
        ----------
        x : numpy array
            State.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        nv = []
        for el in x:
            bee = -el[2]*el[0]+ el[2]*el[3]-el[2]*el[1]
            disc = 0
            for dev in range(0, len(el[4:])):
                disc -= (el[4 + dev]**2)*DISSATISFACTION_COEFFICIENTS[dev]
            new = bee + disc
            nv.append(new)
        return np.array(nv).reshape((-1,1))
    
    def basis_2(self, x):
        """
        Calculating feature transformation as the difference between base and 
        consumption and abs dissatisfaction

        Parameters
        ----------
        x : numpy array
            State.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        nv = []
        for el in x:
            bee = -el[2]*el[0]+ el[2]*el[3]-el[2]*el[1]
            disc = 0
            for dev in range(0, len(el[4:])):
                disc -= (el[4 + dev])*DISSATISFACTION_COEFFICIENTS[dev]
            new = bee + disc
            nv.append(new)
        return np.array(nv).reshape((-1,1))
    
    def basis_3(self, x):
        """
        Calculating feature transformation as only the difference between base 
        and consumption

        Parameters
        ----------
        x : numpy array
            State.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        nv = []
        for el in x:
            bee = -el[2]*el[0]+ el[2]*el[3]-el[2]*el[1]
            new = bee
            nv.append(new)
        return np.array(nv).reshape((-1,1))
    
    def basis_4(self, x):
        """
        Calculating feature transformation as only the difference between base 
        and consumption, but only if its positive

        Parameters
        ----------
        x : numpy array
            State.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        nv = []
        for el in x:
            bee = -el[2]*el[0]+ el[2]*el[3]-el[2]*el[1]
            new = max(0, bee)
            nv.append(new)
        return np.array(nv).reshape((-1,1))
    
    def basis_5(self, x):
        """
        Calculating feature transformation as only the difference between base 
        and consumption, but only if its positive

        Parameters
        ----------
        x : numpy array
            State.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        nv = []
        for el in x:
            bee = -el[2]*el[0]+ el[2]*el[3]-el[2]*el[1]
            disc = 0
            for dev in range(0, len(el[4:])):
                disc -= (el[4 + dev])*DISSATISFACTION_COEFFICIENTS[dev]
            new = max(0, bee) + disc
            nv.append(new)
        return np.array(nv).reshape((-1,1))
    

    def calculate_value(self, traj):
        """
        Calculating approximations of value functions for new arriving samples.
        The final result is V[s0] = (
            V_is, 
            number of samples for averaging
            )
        V_is = list of averaged summed V values for each kernel i. Summed V 
        values are for all trajectory under certain policy pi_k.
        
        Parameters
        ----------
        traj : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        traj = np.array(traj)
        gam = np.ones((traj.shape[0],))*DISCOUNT_RATE**np.arange(traj.shape[0])
        gam = gam.reshape((-1,1))
        vs = []
    
        for m in range(0, 6):
            if (m == 0):
                nv = self.true_test(traj)
            if (m == 1):
                nv = self.basis_1(traj)
            if (m == 2):
                nv = self.basis_2(traj)
            if (m == 3):
                nv = self.basis_3(traj)                
            if (m == 4):
                nv = self.basis_4(traj)
            if (m == 5):
                nv = self.basis_5(traj)
            vs.append((nv*gam).sum(axis=0))

        if round(traj[0].sum(),2) in list(self.V.keys()):
            n = self.V[round(traj[0].sum(),2)][1]
            prev_exp = self.V[round(traj[0].sum(),2)][0]
            val = prev_exp*(n/(n + 1)) + np.array(vs)*1/(n + 1)
            self.V[round(traj[0].sum(),2)] = (val, n + 1)
        else:
            self.V[round(traj[0].sum(),2)] = (np.array(vs), 1)
    
    def get_value(self):
        return self.V

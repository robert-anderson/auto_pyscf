import definition as defn
from pyscf import scf, mcscf
from pyscf.fciqmcscf import kernels
from pyscf.fciqmcscf import FCIQMCCI
from glob import glob
import numpy as np

read_kernel = kernels.read_1step

iiter = len(glob('mo_coeff.*.npy'))
if iiter==0:
    hf = scf.RHF(defn.mol)
    hf.kernel()
    mo_coeff = hf.mo_coeff
    np.save('mo_coeff.0.npy', hf.mo_coeff)
else:
    mo_coeff = np.load('mo_coeff.{}.npy'.format(iiter-1))
    mycas = mcscf.CASSCF(defn.mol, defn.cas, defn.nelecas)
    mycas.fcisolver = FCIQMCCI(defn.mol)
    mycas.fcisolver.mode = 'read rdms'
    mycas.kernel(mo_coeff, _kern=read_kernel)
    np.save('mo_coeff.{}.npy'.format(iiter), mycas.mo_coeff)

mycas = mcscf.CASSCF(defn.mol, defn.cas, defn.nelecas)
mycas.fcisolver = FCIQMCCI(defn.mol)
mycas.fcisolver.mode = 'dump and die'
mycas.kernel(mo_coeff, _kern=kernels.write)

'''

mol = pyscf.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)

myhf = mol.RHF().run()

# 6 orbitals, 8 electrons
mycas = myhf.CASSCF(6, 8).run()
#
# Note this mycas object can also be created using the APIs of mcscf module:
#
# from pyscf import mcscf
# mycas = mcscf.CASSCF(myhf, 6, 8).run()

# Natural occupancy in CAS space, Mulliken population etc.
# See also 00-simple_casci.py for the instruction of the output of analyze()
# method
mycas.verbose = 4
mycas.analyze()




    def mc1step(self, mo_coeff=None, ci0=None, callback=None):
        return self.kernel(mo_coeff, ci0, callback)

    def mc2step(self, mo_coeff=None, ci0=None, callback=None):
        from pyscf.mcscf import mc2step
        return self.kernel(mo_coeff, ci0, callback, mc2step.kernel)
'''

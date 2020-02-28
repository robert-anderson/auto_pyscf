import sys
sys.path.insert(0, '.')
import definition as defn
from pyscf import scf, mcscf
from pyscf.fciqmcscf import kernels
from pyscf.fciqmcscf import FCIQMCCI
from glob import glob
import numpy as np

read_kernel = kernels.read_1step
ncasorb = defn.cas if not isinstance(defn.cas, tuple) else len(defn.cas)

iiter = len(glob('mo_coeff.*.npy'))
if iiter==0:
    hf = scf.RHF(defn.mol).newton() # newton solver might help for FeP
    hf.kernel()
    mo_coeff = hf.mo_coeff
    if isinstance(defn.cas, tuple):
        tmp = mcscf.CASSCF(hf, ncasorb, defn.nelecas)
        mo_coeff = tmp.sort_mo(defn.cas)
        np.save('mo_coeff.0.npy', mo_coeff)
    else:
        np.save('mo_coeff.0.npy', hf.mo_coeff)
else:
    mo_coeff = np.load('mo_coeff.{}.npy'.format(iiter-1))
    mycas = mcscf.CASSCF(defn.mol, ncasorb, defn.nelecas)
    mycas.mo_coeff=mo_coeff
    mycas.fcisolver = FCIQMCCI(defn.mol)
    mycas.fcisolver.mode = 'read rdms'
    mycas.kernel(_kern=read_kernel)
    np.save('mo_coeff.{}.npy'.format(iiter), mycas.mo_coeff)

hf = scf.RHF(defn.mol)
hf.mo_coeff = mo_coeff
mycas = mcscf.CASSCF(hf, ncasorb, defn.nelecas)
mycas.fcisolver = FCIQMCCI(defn.mol)
mycas.fcisolver.mode = 'dump and die'
mycas.kernel(_kern=kernels.write)


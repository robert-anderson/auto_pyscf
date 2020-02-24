import definition as defn
from pyscf import scf, mcscf
from pyscf.fciqmcscf import kernels
from pyscf.fciqmcscf import FCIQMCCI
from glob import glob
import numpy as np

mycas = mcscf.CASSCF(defn.mol, defn.cas, defn.nelecas)
hf = scf.RHF(defn.mol)
hf.kernel()
mycas.kernel(hf.mo_coeff)

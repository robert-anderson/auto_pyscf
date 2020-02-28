import sys
sys.path.insert(0, '.')
import definition as defn
from pyscf import scf, mcscf
from glob import glob
import numpy as np

ncasorb = defn.cas if not isinstance(defn.cas, tuple) else len(defn.cas)

hf = scf.RHF(defn.mol)
hf.kernel()
mycas = mcscf.CASSCF(hf, ncasorb, defn.nelecas)
if isinstance(defn.cas, tuple):
    mo_coeff = mycas.sort_mo(defn.cas)
else:
    mo_coeff = hf.mo_coeff
mycas.kernel(mo_coeff)

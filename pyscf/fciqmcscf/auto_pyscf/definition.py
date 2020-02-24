import pyscf

mol = pyscf.M(
    atom = 'O 0 0 0; O 0 0 1.2',
    basis = 'ccpvdz',
    spin = 2)

nelecas = 8
cas = 6

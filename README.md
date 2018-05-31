# dstpy3


Algorithms to calculate the Direct Scattering Transform [1] (aka Nonlinear Fourier Transform) for the Nonlinear Schroedinger Equation (NLSE).

It includes implementations of several algorithms presented e.g. in [2-5], mostly for educational use.

## Requirements: 

- on a fundamental level, a decent version of python with numpy and scipy should be sufficient to start (though python vanilla algorithms are quite slow)
- a fast fortran library for the 'Transfer Matrix Method' and Runge-Kutta-4 integration in Double and Quad precision are available. Fortran compiler and working f2py-script are necessary
- as for certain problems, numerical issues may arise, also an arbitray precision version implemented in python-mpmath is implemented (extra slow!)

## minimal example (plain python)

```
from dstpy3 import *
import numpy as np
from isttools import *   # isttools.py should be import seperately ... is located in dstpy folder

gamma = 1
beta2 = -1
tvec = np.linspace(-20,20, 512)
field = 2.3 / np.cosh(tvec)
dob = DSTObj(field, tvec, beta2, gamma, t0scaleext=1.0)
ews = evsearch2(dob, methodite='RK4', methodspec='TM')  # use vanilla python    Transfer Matrix for spectrum, RK4 for solitons
#ews = evsearch2(dob, methodite='RK4F', methodspec='TMF')  # fortran version    
#ews = evsearch2(dob, methodite='RK4FQ', methodspec='TMFQ')  # fortran quad prec version   

print('CONV? {:d}  Ediff {:.3e}  Esol {:.3e} Espec {:.3e}'.format(ews['converged'],ews['E_diff'], ews['E_sol'], ews['E_spec']))
for i in range(len(ews['evals'])):
    print("{:d} Re: {:.3e}  Im: {:.3e}".format(i, np.real(ews['evals'])[i],np.imag(ews['evals'])[i]))
    
(output:)
CONV? 1  Ediff 4.552e-04  Esol 2.600e+00 Espec 4.500e-02
0 Re: 3.403e-26  Im: 7.999e-01
1 Re: 6.248e-21  Im: 1.800e+00
```


## references

[1] A. Shabat and V. Zakharov. ”Exact theory of two-dimensional self-focusing and one-dimensional
self-modulation of waves in nonlinear media.” Soviet physics JETP 34, 62 (1972).

[2] G. Boffetta, A. Osborne, Journal of Computational Physics 102, 252 (1992)

[3] M. I. Yousefi, and F. R. Kschischang. ”Information transmission using the nonlinear Fourier trans-
form, Part I: Mathematical tools.” IEEE Transactions on Information Theory 60 (2014): 4312-4328.

[4] M. I. Yousefi, and F. R. Kschischang. ”Information transmission using the nonlinear Fourier trans-
form, Part II: Numerical methods.” IEEE Transactions on Information Theory 60 (2014): 4329-4345.

[5] M. I. Yousefi, and F. R. Kschischang.”Information transmission using the nonlinear Fourier trans-
form, Part III: Spectrum modulation.” IEEE Transactions on Information Theory 60 (2014): 4346-
4369.

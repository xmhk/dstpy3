# dstpy3


Algorithms to calculate the Direct Scattering Transform [1] (aka Nonlinear Fourier Transform) for the Nonlinear Schroedinger Equation (NLSE).

It includes implementations of several algorithms presented e.g. in [2-5], mostly for educational use.

Requirements: 

- on a fundamental level, a decent version of python with numpy and scipy should be sufficient to start (though python vanilla algorithms are quite slow)
- a fast fortran library for the 'Transfer Matrix Method' and Runge-Kutta-4 integration in Double and Quad precision are available. Fortran compiler and working f2py-script are necessary
- as for certain problems, numerical issues may arise, also an arbitray precision version implemented in python-mpmath is implemented (extra slow!)



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

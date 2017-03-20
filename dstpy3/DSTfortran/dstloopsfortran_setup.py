from __future__ import division, absolute_import, print_function

from numpy.distutils.core import Extension


ext1 = Extension(name = 'dstloopsfortran',
                 sources = ['dstloopsfortran.f90'])

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name = 'dstloopsfortran',
          description       = "F2PY fortran loops for DST",
          author            = "xm",
          author_email      = "xm",
          ext_modules = [ext1]
          )
# End of setup_example.py
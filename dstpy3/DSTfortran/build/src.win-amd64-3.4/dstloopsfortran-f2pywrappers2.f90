!     -*- f90 -*-
!     This file is autogenerated with f2py (version:2)
!     It contains Fortran 90 wrappers to fortran functions.

      subroutine f2pywrap_dstloopsfort_loopf_diff (dx, l, q, zeta, lq, l&
     &zeta, adash, bdash, addash, bddash, f2py_q_d0, f2py_zeta_d0, f2py_&
     &adash_d0, f2py_bdash_d0, f2py_addash_d0, f2py_bddash_d0)
      use dstloopsfort, only : loopf_diff
      real(kind=8) dx
      real(kind=8) l
      integer lq
      integer lzeta
      integer f2py_q_d0
      integer f2py_zeta_d0
      integer f2py_adash_d0
      integer f2py_bdash_d0
      integer f2py_addash_d0
      integer f2py_bddash_d0
      complex(kind=8) q(f2py_q_d0)
      complex(kind=8) zeta(f2py_zeta_d0)
      complex(kind=8) adash(f2py_adash_d0)
      complex(kind=8) bdash(f2py_bdash_d0)
      complex(kind=8) addash(f2py_addash_d0)
      complex(kind=8) bddash(f2py_bddash_d0)
      call loopf_diff(dx, l, q, zeta, lq, lzeta, adash, bdash, addash, b&
     &ddash)
      end subroutine f2pywrap_dstloopsfort_loopf_diff
      subroutine f2pywrap_dstloopsfort_loopf (dx, l, q, zeta, lq, lzeta,&
     & adash, bdash, f2py_q_d0, f2py_zeta_d0, f2py_adash_d0, f2py_bdash_&
     &d0)
      use dstloopsfort, only : loopf
      real(kind=8) dx
      real(kind=8) l
      integer lq
      integer lzeta
      integer f2py_q_d0
      integer f2py_zeta_d0
      integer f2py_adash_d0
      integer f2py_bdash_d0
      complex(kind=8) q(f2py_q_d0)
      complex(kind=8) zeta(f2py_zeta_d0)
      complex(kind=8) adash(f2py_adash_d0)
      complex(kind=8) bdash(f2py_bdash_d0)
      call loopf(dx, l, q, zeta, lq, lzeta, adash, bdash)
      end subroutine f2pywrap_dstloopsfort_loopf
      
      subroutine f2pyinitdstloopsfort(f2pysetupfunc)
      interface 
      subroutine f2pywrap_dstloopsfort_loopf_diff (dx, l, q, zeta, lq, l&
     &zeta, adash, bdash, addash, bddash, f2py_q_d0, f2py_zeta_d0, f2py_&
     &adash_d0, f2py_bdash_d0, f2py_addash_d0, f2py_bddash_d0)
      real(kind=8) dx
      real(kind=8) l
      integer lq
      integer lzeta
      integer f2py_q_d0
      integer f2py_zeta_d0
      integer f2py_adash_d0
      integer f2py_bdash_d0
      integer f2py_addash_d0
      integer f2py_bddash_d0
      complex(kind=8) q(f2py_q_d0)
      complex(kind=8) zeta(f2py_zeta_d0)
      complex(kind=8) adash(f2py_adash_d0)
      complex(kind=8) bdash(f2py_bdash_d0)
      complex(kind=8) addash(f2py_addash_d0)
      complex(kind=8) bddash(f2py_bddash_d0)
      end subroutine f2pywrap_dstloopsfort_loopf_diff 
      subroutine f2pywrap_dstloopsfort_loopf (dx, l, q, zeta, lq, lzeta,&
     & adash, bdash, f2py_q_d0, f2py_zeta_d0, f2py_adash_d0, f2py_bdash_&
     &d0)
      real(kind=8) dx
      real(kind=8) l
      integer lq
      integer lzeta
      integer f2py_q_d0
      integer f2py_zeta_d0
      integer f2py_adash_d0
      integer f2py_bdash_d0
      complex(kind=8) q(f2py_q_d0)
      complex(kind=8) zeta(f2py_zeta_d0)
      complex(kind=8) adash(f2py_adash_d0)
      complex(kind=8) bdash(f2py_bdash_d0)
      end subroutine f2pywrap_dstloopsfort_loopf
      end interface
      external f2pysetupfunc
      call f2pysetupfunc(f2pywrap_dstloopsfort_loopf_diff,f2pywrap_dstlo&
     &opsfort_loopf)
      end subroutine f2pyinitdstloopsfort


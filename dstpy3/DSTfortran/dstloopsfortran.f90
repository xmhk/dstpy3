module dstloopsfort

contains


  ! ---------------------------------------------------------------------------------
  !
  !   RK4 algos with and without derivatives
  !
  ! ---------------------------------------------------------------------------------

  ! RK4 with double precision
  subroutine loopf( dx, l, q, zeta, lq, lzeta, adash, bdash)
    implicit none
    integer, parameter :: hprec = 8
    complex(kind=hprec)  ::  plusi = cmplx(  0.0,   1.0)
    complex(kind=hprec)  :: minusi = cmplx(  0.0, - 1.0)
    complex(kind=hprec)  :: minus1 = cmplx( -1.0,   0.0)
    complex(kind=hprec)  ::    nll = cmplx(  0.0,   0.0)
    complex(kind=hprec), dimension(2,2) :: pk(2,2)
    real(kind=8) :: dx, l
    integer :: i,ii, lzeta, lq
    complex(kind=8) ::q(:), adash(:), bdash(:),zeta(:)
    complex(kind=8) :: aktzeta
    complex(kind=hprec), dimension(:,:), allocatable :: v(:,:)
    complex(kind=hprec), dimension(2) :: k1, k2, k3, k4
    pk = reshape( (/nll, nll, nll,nll, nll /),(/2,2/))
    allocate(v( 0:lq, 2))
    do i=1,lzeta
        aktzeta = zeta(i)
        pk(1,1) = minusi * aktzeta; pk(2,2) = plusi * aktzeta;
        v(1,1) =             exp( plusi * aktzeta * l);      v(1,2) = nll
        do ii = 1, lq-1
          pk(2,1) = minus1* conjg( q(ii)); pk(1,2) = q(ii)
          k1 = matmul( pk , v(ii,:))
          k2 = matmul( pk, v(ii,:) + dx / 2.0 * k1)
          k3 = matmul( pk, v(ii,:) + dx / 2.0 * k2)
          k4 = matmul( pk, v(ii,:) +  dx *k3)
          v(ii+1,:) = v(ii,:) +  dx * 1./6. *( k1 + 2*k2 + 2*k3 + k4)
        end do ! q loop for aktzeta
        adash(i) = v(ii,1) * exp(plusi * aktzeta * l)
        bdash(i) = v(ii,2) * exp(minusi * aktzeta * l)
      end do !zetaloop
      deallocate(v)
    end subroutine loopf


  ! RK4 with double precision
  subroutine loopf_diff( dx, l, q, zeta, lq, lzeta, adash, bdash, addash, bddash )
      implicit none
      integer, parameter :: hprec = 8
      complex(kind=hprec)  ::  plusi = cmplx(  0.0,   1.0)
      complex(kind=hprec)  :: minusi = cmplx(  0.0, - 1.0)
      complex(kind=hprec)  :: minus1 = cmplx( -1.0,   0.0)
      complex(kind=hprec)  ::    nll = cmplx(  0.0,   0.0)
      complex(kind=hprec), dimension(4,4) :: pk(4,4)
      real(kind=8) :: dx, l
      integer :: i,ii, lzeta, lq
      complex(kind=8) ::q(:), adash(:), bdash(:), addash(:), bddash(:),zeta(:)
      complex(kind=8) :: aktzeta, aktq
      complex(kind=hprec), dimension(:,:), allocatable :: v(:,:)
      complex(kind=hprec), dimension(4) :: k1, k2, k3, k4
      !initialize pk
      pk = reshape( (/nll, nll, nll,nll, nll, nll, nll,nll, nll, nll, nll,nll, nll, nll, nll,nll /),(/4,4/))
      ! these values of pk are constant
      pk(3,1) = minusi;     pk(4,1) = nll ;    pk(3,2) = nll ;    pk(4,2) = plusi
      pk(1,3) = nll ;       pk(2,3) = nll   ;  pk(1,4) = nll ;    pk(2,4) = nll
      allocate(v( 0:lq, 4))
      do i=1,lzeta
          aktzeta = zeta(i)
          ! set values of pk that only depend on aktzeta
          pk(1,1) = minusi * aktzeta; pk(2,2) = plusi * aktzeta;
          pk(3,3) = minusi * aktzeta; pk(4,4) = plusi * aktzeta
          ! initialize first elements of v
          v(1,1) =             exp( plusi * aktzeta * l);      v(1,2) = nll
          v(1,3) = plusi * l * exp( plusi * aktzeta * l);      v(1,4) = nll
          do ii = 1, lq-1
            aktq = q(ii);
            pk(2,1) = minus1* conjg( aktq); pk(1,2) = aktq
            pk(4,3) = minus1 * conjg(aktq); pk(3,4) = aktq
            k1 = matmul( pk , v(ii,:))
            k2 = matmul( pk, v(ii,:) + dx/2.0 * k1)
            k3 = matmul( pk, v(ii,:) + dx/2.0 * k2)
            k4 = matmul( pk, v(ii,:) +  dx *k3)
            v(ii+1,:) = v(ii,:) +  dx * 1./6. *( k1 + 2*k2 + 2*k3 + k4)
          end do ! q loop for aktzeta
          adash(i) = v(ii,1) * exp(plusi * aktzeta * l)
          bdash(i) = v(ii,2) * exp(minusi * aktzeta * l)
          addash(i) = (v(ii,3) + plusi * l * v(ii,1)) * exp(plusi* aktzeta *l)
          bddash(i) = nll
      end do !zetaloop
      deallocate(v)
    end subroutine loopf_diff


   ! RK4 with quad precision for intermediate steps
    subroutine loopf_qp( dx, l, q, zeta, lq, lzeta, adash, bdash)
        implicit none
        integer, parameter :: hprec = 16
        complex(kind=hprec)  ::  plusi = cmplx(  0.0,   1.0)
        complex(kind=hprec)  :: minusi = cmplx(  0.0, - 1.0)
        complex(kind=hprec)  :: minus1 = cmplx( -1.0,   0.0)
        complex(kind=hprec)  ::    nll = cmplx(  0.0,   0.0)
        complex(kind=hprec), dimension(2,2) :: pk(2,2)
        real(kind=8) :: dx, l
        integer :: i,ii, lzeta, lq
        complex(kind=8) ::q(:), adash(:), bdash(:),zeta(:)
        complex(kind=8) :: aktzeta
        complex(kind=hprec), dimension(:,:), allocatable :: v(:,:)
        complex(kind=hprec), dimension(2) :: k1, k2, k3, k4
        pk = reshape( (/nll, nll, nll,nll, nll /),(/2,2/))
        allocate(v( 0:lq, 2))
        do i=1,lzeta
            aktzeta = zeta(i)
            pk(1,1) = minusi * aktzeta; pk(2,2) = plusi * aktzeta;
            v(1,1) =             exp( plusi * aktzeta * l);      v(1,2) = nll
            do ii = 1, lq-1
              pk(2,1) = minus1* conjg( q(ii)); pk(1,2) = q(ii)
              k1 = matmul( pk , v(ii,:))
              k2 = matmul( pk, v(ii,:) + dx / 2.0 * k1)
              k3 = matmul( pk, v(ii,:) + dx / 2.0 * k2)
              k4 = matmul( pk, v(ii,:) +  dx *k3)
              v(ii+1,:) = v(ii,:) +  dx * 1./6. *( k1 + 2*k2 + 2*k3 + k4)
            end do ! q loop for aktzeta
            adash(i) = v(ii,1) * exp(plusi * aktzeta * l)
            bdash(i) = v(ii,2) * exp(minusi * aktzeta * l)
        end do !zetaloop
        deallocate(v)
    end subroutine loopf_qp





! with internally quad precision
  subroutine loopf_diff_qp( dx, l, q, zeta, lq, lzeta, adash, bdash, addash, bddash )
      implicit none
      integer, parameter :: hprec = 16
      complex(kind=hprec)  ::  plusi = cmplx(  0.0,   1.0)
      complex(kind=hprec)  :: minusi = cmplx(  0.0, - 1.0)
      complex(kind=hprec)  :: minus1 = cmplx( -1.0,   0.0)
      complex(kind=hprec)  ::    nll = cmplx(  0.0,   0.0)
      complex(kind=hprec), dimension(4,4) :: pk(4,4)
      real(kind=8) :: dx, l
      integer :: i,ii, lzeta, lq
      complex(kind=8) ::q(:), adash(:), bdash(:), addash(:), bddash(:),zeta(:)
      complex(kind=8) :: aktzeta, aktq
      complex(kind=hprec), dimension(:,:), allocatable :: v(:,:)
      complex(kind=hprec), dimension(4) :: k1, k2, k3, k4
      !initialize pk
      pk = reshape( (/nll, nll, nll,nll, nll, nll, nll,nll, nll, nll, nll,nll, nll, nll, nll,nll /),(/4,4/))
      ! these values of pk are constant
      pk(3,1) = minusi;     pk(4,1) = nll ;    pk(3,2) = nll ;    pk(4,2) = plusi
      pk(1,3) = nll ;       pk(2,3) = nll   ;  pk(1,4) = nll ;    pk(2,4) = nll
      allocate(v( 0:lq, 4))
      do i=1,lzeta
          aktzeta = zeta(i)
          ! set values of pk that only depend on aktzeta
          pk(1,1) = minusi * aktzeta; pk(2,2) = plusi * aktzeta;
          pk(3,3) = minusi * aktzeta; pk(4,4) = plusi * aktzeta
          ! initialize first elements of v
          v(1,1) =             exp( plusi * aktzeta * l);      v(1,2) = nll
          v(1,3) = plusi * l * exp( plusi * aktzeta * l);      v(1,4) = nll
          do ii = 1, lq-1
            aktq = q(ii);
            pk(2,1) = minus1* conjg( aktq); pk(1,2) = aktq
            pk(4,3) = minus1 * conjg(aktq); pk(3,4) = aktq
            k1 = matmul( pk , v(ii,:))
            k2 = matmul( pk, v(ii,:) + dx/2.0 * k1)
            k3 = matmul( pk, v(ii,:) + dx/2.0 * k2)
            k4 = matmul( pk, v(ii,:) +  dx *k3)
            v(ii+1,:) = v(ii,:) +  dx * 1./6. *( k1 + 2*k2 + 2*k3 + k4)
          end do ! q loop for aktzeta
          adash(i) = v(ii,1) * exp(plusi * aktzeta * l)
          bdash(i) = v(ii,2) * exp(minusi * aktzeta * l)
          addash(i) = (v(ii,3) + plusi * l * v(ii,1)) * exp(plusi* aktzeta *l)
          bddash(i) = nll
      end do !zetaloop
      deallocate(v)
  end subroutine loopf_diff_qp

    ! ---------------------------------------------------------------------------------
    !
    !   Transfer matrix algos with and without derivatives
    !
    ! ---------------------------------------------------------------------------------

    ! TM with double precision
    subroutine loopf_tm( dx, l, q, zeta, lq, lzeta, adash, bdash)
        implicit none
        integer, parameter :: hprec = 8
        complex(kind=hprec)  ::  plusi = cmplx(  0.0,   1.0)
        complex(kind=hprec)  :: minusi = cmplx(  0.0, - 1.0)
        complex(kind=hprec)  :: minus1 = cmplx( -1.0,   0.0)
        complex(kind=hprec)  ::    nll = cmplx(  0.0,   0.0)
        complex(kind=hprec)  ::    eins = cmplx(  1.0,   0.0)
        complex(kind=hprec), dimension(2,2) :: T(2,2)
        complex(kind=hprec), dimension(2,2) :: S(2,2)
        real(kind=8) :: dx, l
        integer :: i,ii, lzeta, lq
        complex(kind=8) ::q(:), adash(:), bdash(:), zeta(:)
        complex(kind=hprec) :: k, coshkdxm, sinhkdxm
        complex(kind=hprec) :: U11,U12,U21,U22
        do i=1,lzeta
           S = reshape( (/eins, nll, nll,eins /),(/2,2/))
           ii = lq ! or  lq-1 ???
           do while (ii>0)
             k= sqrt( minus1 * abs(q(ii))**2 - zeta(i)**2)
             coshkdxm = cosh(k*dx)
             sinhkdxm = sinh(k*dx)
             U11 = coshkdxm  +  minusi * zeta(i) / k * sinhkdxm
             U12 = q(ii)/k * sinhkdxm
             U21 = minus1 * conjg(q(ii)) / k * sinhkdxm
             U22 = coshkdxm + plusi * zeta(i) / k * sinhkdxm
             T = reshape( (/U11, U12,&
                            U21, U22/), (/2,2/))
             S = matmul( S, T)
             ii = ii -1
           end do
           ! remember: fortran indexing: C,R ...
           adash(i) = S(1,1) * exp( 2 * plusi * zeta(i) * l)
           bdash(i) = S(1,2)
        end do !zetaloop
      end subroutine loopf_tm

      ! TM with double precision
      subroutine loopf_tm_diff( dx, l, q, zeta, lq, lzeta, adash, bdash, addash, bddash )
          implicit none
          integer, parameter :: hprec = 8
          complex(kind=hprec)  ::  plusi = cmplx(  0.0,   1.0)
          complex(kind=hprec)  :: minusi = cmplx(  0.0, - 1.0)
          complex(kind=hprec)  :: minus1 = cmplx( -1.0,   0.0)
          complex(kind=hprec)  ::    nll = cmplx(  0.0,   0.0)
          complex(kind=hprec)  ::    eins = cmplx(  1.0,   0.0)
          complex(kind=hprec), dimension(4,4) :: T(4,4)
          complex(kind=hprec), dimension(4,4) :: S(4,4)

          real(kind=8) :: dx, l
          integer :: i,ii, lzeta, lq
          complex(kind=8) ::q(:), adash(:), bdash(:), addash(:), bddash(:),zeta(:)
          complex(kind=hprec) :: k, coshkdxm, sinhkdxm
          complex(kind=hprec) :: U11,U12,U21,U22, UD11,UD12,UD21,UD22
          do i=1,lzeta
             S = reshape( (/eins, nll, nll,nll,   nll, eins, nll,nll,    &
                            nll, nll, eins, nll,  nll, nll, nll,eins /),(/4,4/))
             ii = lq ! or  lq-1 ???
             do while (ii>0)
               k= sqrt( minus1 * abs(q(ii))**2 - zeta(i)**2)
               coshkdxm = cosh(k*dx)
               sinhkdxm = sinh(k*dx)
               U11 = coshkdxm  +  minusi * zeta(i) / k * sinhkdxm
               U12 = q(ii)/k * sinhkdxm
               U21 = minus1 * conjg(q(ii)) / k * sinhkdxm
               U22 = coshkdxm + plusi * zeta(i) / k * sinhkdxm

               UD11 = plusi * dx * zeta(i)**2 / k**2 * coshkdxm &
                          -(zeta(i) * dx + plusi + plusi * zeta(i)**2/k**2) * sinhkdxm/k
               UD12 = minus1 * q(ii) * zeta(i) / k**2 * (dx * coshkdxm - sinhkdxm/k)
               UD21 = conjg(q(ii)) * zeta(i) / k**2 * (dx * coshkdxm - sinhkdxm/k)
               UD22 = minusi *  dx * zeta(i)**2 / k**2 * coshkdxm &
                        -(zeta(i) * dx - plusi - plusi * zeta(i)**2/k**2) * sinhkdxm/k
               T = reshape( (/U11, U12, nll,nll,&
                              U21, U22, nll, nll,&
                              UD11, UD12, U11, U12,&
                              UD21, UD22, U21, U22/), (/4,4/))
               S = matmul( S, T)
               ii = ii -1
             end do
             ! remember: fortran indexing: C,R ...
             adash(i) = S(1,1) * exp( 2 * plusi * zeta(i) * l)
             bdash(i) = S(1,2)
             addash(i)=( S(1,3) + plusi * l *(S(1,1) + S(3,3)) )* exp(2.0 * plusi * zeta(i) * l)
             bddash(i) = S(1,4) + plusi * l * (S(3,4)-S(1,2))
          end do !zetaloop
        end subroutine loopf_tm_diff


     ! TM with quad precision for intermediate steps
      subroutine loopf_tm_quad( dx, l, q, zeta, lq, lzeta, adash, bdash)
          implicit none
          integer, parameter :: hprec = 16
          complex(kind=hprec)  ::  plusi = cmplx(  0.0,   1.0)
          complex(kind=hprec)  :: minusi = cmplx(  0.0, - 1.0)
          complex(kind=hprec)  :: minus1 = cmplx( -1.0,   0.0)
          complex(kind=hprec)  ::    nll = cmplx(  0.0,   0.0)
          complex(kind=hprec)  ::    eins = cmplx(  1.0,   0.0)
          complex(kind=hprec), dimension(2,2) :: T(2,2)
          complex(kind=hprec), dimension(2,2) :: S(2,2)
          real(kind=8) :: dx, l
          integer :: i,ii, lzeta, lq
          complex(kind=8) ::q(:), adash(:), bdash(:), zeta(:)
          complex(kind=hprec) :: k, coshkdxm, sinhkdxm
          complex(kind=hprec) :: U11,U12,U21,U22
          do i=1,lzeta
             S = reshape( (/eins, nll, nll,eins /),(/2,2/))
             ii = lq ! or  lq-1 ???
             do while (ii>0)
               k= sqrt( minus1 * abs(q(ii))**2 - zeta(i)**2)
               coshkdxm = cosh(k*dx)
               sinhkdxm = sinh(k*dx)
               U11 = coshkdxm  +  minusi * zeta(i) / k * sinhkdxm
               U12 = q(ii)/k * sinhkdxm
               U21 = minus1 * conjg(q(ii)) / k * sinhkdxm
               U22 = coshkdxm + plusi * zeta(i) / k * sinhkdxm
               T = reshape( (/U11, U12,&
                              U21, U22/), (/2,2/))
               S = matmul( S, T)
               ii = ii -1
             end do
             ! remember: fortran indexing: C,R ...
             adash(i) = S(1,1) * exp( 2 * plusi * zeta(i) * l)
             bdash(i) = S(1,2)
          end do !zetaloop
        end subroutine loopf_tm_quad


     ! TM with quad precision for intermediate steps
      subroutine loopf_tm_diff_quad( dx, l, q, zeta, lq, lzeta, adash, bdash, addash, bddash )
          implicit none
          integer, parameter :: hprec = 16
          complex(kind=hprec)  ::  plusi = cmplx(  0.0,   1.0)
          complex(kind=hprec)  :: minusi = cmplx(  0.0, - 1.0)
          complex(kind=hprec)  :: minus1 = cmplx( -1.0,   0.0)
          complex(kind=hprec)  ::    nll = cmplx(  0.0,   0.0)
          complex(kind=hprec)  ::    eins = cmplx(  1.0,   0.0)
          complex(kind=hprec), dimension(4,4) :: T(4,4)
          complex(kind=hprec), dimension(4,4) :: S(4,4)

          real(kind=8) :: dx, l
          integer :: i,ii, lzeta, lq
          complex(kind=8) ::q(:), adash(:), bdash(:), addash(:), bddash(:),zeta(:)
          complex(kind=hprec) :: k, coshkdxm, sinhkdxm
          complex(kind=hprec) :: U11,U12,U21,U22, UD11,UD12,UD21,UD22
          do i=1,lzeta
             S = reshape( (/eins, nll, nll,nll,   nll, eins, nll,nll,    &
                            nll, nll, eins, nll,  nll, nll, nll,eins /),(/4,4/))
             ii = lq ! or  lq-1 ???
             do while (ii>0)
               k= sqrt( minus1 * abs(q(ii))**2 - zeta(i)**2)
               coshkdxm = cosh(k*dx)
               sinhkdxm = sinh(k*dx)
               U11 = coshkdxm  +  minusi * zeta(i) / k * sinhkdxm
               U12 = q(ii)/k * sinhkdxm
               U21 = minus1 * conjg(q(ii)) / k * sinhkdxm
               U22 = coshkdxm + plusi * zeta(i) / k * sinhkdxm

               UD11 = plusi * dx * zeta(i)**2 / k**2 * coshkdxm &
                          -(zeta(i) * dx + plusi + plusi * zeta(i)**2/k**2) * sinhkdxm/k
               UD12 = minus1 * q(ii) * zeta(i) / k**2 * (dx * coshkdxm - sinhkdxm/k)
               UD21 = conjg(q(ii)) * zeta(i) / k**2 * (dx * coshkdxm - sinhkdxm/k)
               UD22 = minusi *  dx * zeta(i)**2 / k**2 * coshkdxm &
                        -(zeta(i) * dx - plusi - plusi * zeta(i)**2/k**2) * sinhkdxm/k
               T = reshape( (/U11, U12, nll,nll,&
                              U21, U22, nll, nll,&
                              UD11, UD12, U11, U12,&
                              UD21, UD22, U21, U22/), (/4,4/))
               S = matmul( S, T)
               ii = ii -1
             end do
             ! remember: fortran indexing: C,R ...
             adash(i) = S(1,1) * exp( 2 * plusi * zeta(i) * l)
             bdash(i) = S(1,2)
             addash(i)=( S(1,3) + plusi * l *(S(1,1) + S(3,3)) )* exp(2.0 * plusi * zeta(i) * l)
             bddash(i) = S(1,4) + plusi * l * (S(3,4)-S(1,2))
          end do !zetaloop
        end subroutine loopf_tm_diff_quad

end module dstloopsfort

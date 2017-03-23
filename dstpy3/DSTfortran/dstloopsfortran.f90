module dstloopsfort

contains

subroutine loopf_diff( dx, l, q, zeta, lq, lzeta, adash, bdash, addash, bddash )
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
        aktq = q(1)
        pk(2,1) = minus1* conjg( aktq); pk(1,2) = aktq
        pk(4,3) = minus1 * conjg(aktq); pk(3,4) = aktq
        v(2,:) = v(1,:) + dx * matmul ( pk , v(1,:))
        ! integrate other elements using RK4
        do ii = 1, lq-2
          aktq = q(ii);
          pk(2,1) = minus1* conjg( aktq); pk(1,2) = aktq
          pk(4,3) = minus1 * conjg(aktq); pk(3,4) = aktq
          k1 = matmul( pk , v(ii,:))
          aktq = q(ii+1);
          pk(2,1) = minus1* conjg( aktq); pk(1,2) = aktq
          pk(4,3) = minus1 * conjg(aktq); pk(3,4) = aktq
          k2 = matmul( pk, v(ii,:) + dx * k1)
          k3 = matmul( pk, v(ii,:) + dx * k2)
          aktq = q(ii+2);
          pk(2,1) = minus1* conjg( aktq); pk(1,2) = aktq
          pk(4,3) = minus1 * conjg(aktq); pk(3,4) = aktq
          k4 = matmul( pk, v(ii,:) + 2 * dx *k3)
          v(ii+2,:) = v(ii,:) +  dx * 1./3. *( k1 + 2*k2 + 2*k3 + k4)
        end do ! q loop for aktzeta
        adash(i) = v(ii+1,1) * exp(plusi * aktzeta * l)
        bdash(i) = v(ii+1,2) * exp(minusi * aktzeta * l)
        addash(i) = (v( ii+1,3) + plusi * L * v(ii+1,1)) * exp(plusi* aktzeta *l)
        bddash(i) = nll
    end do !zetaloop
    deallocate(v)
end subroutine loopf_diff

subroutine loopf( dx, l, q, zeta, lq, lzeta, adash, bdash)
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
end subroutine loopf


end module dstloopsfort

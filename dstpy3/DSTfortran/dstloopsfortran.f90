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


subroutine loopf( dx, l, q, zeta, lq, lzeta, adash, bdash )
    implicit none
    integer, parameter :: qp = 16  !quad precision 
    integer, parameter :: dp = 8   
    integer :: i,ii
    integer :: lzeta, lq
    real(kind = 8)   :: dx, L    
   
    complex(kind = 8) ::q(:), adash(:), bdash(:)
    complex(kind = 8) ::zeta(:) 
    
    ! internally use vectors with quad precision
    complex(kind=16), dimension(:), allocatable :: a,b, v1, v2   
    complex(kind=16) ::  k11, k12, k21, k22, k31, k32, k41, k42    
    complex(kind=16) :: aktzeta, qii, qii1, qii2, q0, v1ii, v2ii    
    complex(kind=16), dimension(4) :: p0, fpii, fpii1, fpii2
    complex(kind=16)  :: minusi = cmplx(0.0,-1.0)
    complex(kind=16)  :: plusi = cmplx(0.0,1.0)    ! 
    !  allocate memory
    !
    allocate(a(0:lzeta))
    allocate(b(0:lzeta))
    allocate(v1(0:lq))
    allocate(v2(0:lq))  
    ! loop over zetas. 
    do i=1,lzeta        
        aktzeta = zeta(i)      
        q0 = q(1)
        v1(0) = exp( minusi * aktzeta *cmplx(-1.0,0) * l)
        v2(0) = cmplx(0,0)
        p0(1:4) = (/  minusi * aktzeta, q0, -1.0 * conjg(q0), plusi * aktzeta /)        
        v1(1) = v1(0) + dx * (  p0(1) * v1(0) + p0(2) * v2(0))
        v2(1) = v2(0) + dx * (  p0(3) * v1(0) + p0(4) * v2(0))     
        do ii=0,lq-2
            qii = q(ii)
            qii1 = q(ii+1)
            qii2 = q(ii+2)
            v1ii = v1(ii)
            v2ii = v2(ii)
            fpii =  (/ minusi * aktzeta,    qii     ,-1.0 * conjg(qii),    plusi * aktzeta /)
            fpii1 = (/ minusi * aktzeta,    qii1    ,-1.0 * conjg(qii1),   plusi * aktzeta /)
            fpii2 = (/ minusi * aktzeta,    qii2    ,-1.0 * conjg(qii2),   plusi * aktzeta /)            
            k11 = fpii(1) * v1ii + fpii(2) * v2ii
            k12 = fpii(3) * v1ii + fpii(4) * v2ii            
            k21 = fpii1(1) * (v1ii + dx*k11)  + fpii1(2) * (v2ii + dx*k12)
            k22 = fpii1(3) * (v1ii + dx*k11)  + fpii1(4) * (v2ii + dx*k12)            
            k31 = fpii1(1) * (v1ii + dx*k21)  + fpii1(2) * (v2ii + dx*k22)
            k32 = fpii1(3) * (v1ii + dx*k21)  + fpii1(4) * (v2ii + dx*k22)            
            k41 = fpii2(1) * (v1ii + 2*dx*k31)  + fpii2(2) * (v2ii + 2*dx*k32)
            k42 = fpii2(3) * (v1ii + 2*dx*k31)  + fpii2(4) * (v2ii + 2*dx*k32)        
            v1(ii+2) = v1ii + 2 * dx * 1./6. * ( k11 + 2*k21 + 2*k31 + k41)  
            v2(ii+2) = v2ii + 2 * dx * 1./6. *  (k12 + 2*k22 + 2*k32 + k42)              
        end do  ! attention: ii increments 1 after loop is finished
        a(i) = v1(ii+1) * exp( plusi * aktzeta * l)
        b(i) = v2(ii+1) * exp( plusi * aktzeta * l)        
    end do  
    do i = 1,lzeta        
        adash(i) = a(i)
        bdash(i) = b(i)       
    end do   
    deallocate(a)
    deallocate(b)
    deallocate(v1)
    deallocate(v2)
end subroutine loopf

end module dstloopsfort


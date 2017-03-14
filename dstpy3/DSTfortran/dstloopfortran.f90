subroutine dstloopf( dx, l, q, zeta, lq, lzeta, adash, bdash )
    implicit none
    !
    ! f2py -c dstmod.f90 -m dstf
    ! double and quart
    integer, parameter :: qp = 16  !selected_real_kind(33, 4931) ! 16
    integer, parameter :: dp = 8   !selected_real_kind(15, 307)  ! 8
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
    complex(kind=16)  :: plusi = cmplx(0.0,1.0)
    ! 
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
        !write(*,*) " "
        !write(*,*) " "
        !write(*,*) " "
        !write( *,*)"ii", ii
        !write(*,*) "qii", qii
        !write(*,*) "fpiii", fpii
        !write(*,*) "fpiii1", fpii1
        !write(*,*) "fpiii2", fpii2
        !write(*,*) "k11, k12", k11, k12
        !write(*,*) "k21, k22", k21, k22
        !write(*,*) "k31, k32", k31, k32
        !write(*,*) "k41, k42", k41, k42
        !write(*,*) "dx", dx       
        !write(*,*) "aktzeta" ,aktzeta
        !write(*,*) "l",l       
        !write(*,*) "v1(ii+1)",v1(ii+1)
        !write(*,*) "aktzeta * l", aktzeta * l
        !write(*,*) " "
        a(i) = v1(ii+1) * exp( plusi * aktzeta * l)
        b(i) = v2(ii+1) * exp( plusi * aktzeta * l)
       
        
    end do
    ! assign values of a and b to adash and bdash  ...
    ! possible problems: conversion von complex(16) to complex(8) numbers
    ! ... seems to work on this machine :)
    do i = 1,lzeta        
        adash(i) = a(i)
        bdash(i) = b(i)
        !write(*,*) i, a(i), adash(i)
    end do
    
    !
    ! deallocale mem
    !
    deallocate(a)
    deallocate(b)
    deallocate(v1)
    deallocate(v2)
end subroutine dstloopf

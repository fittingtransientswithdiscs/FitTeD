program wrapper

! gfortran tdedisk.f90
  
  implicit none
  integer ne,i,j,jmax,nr
  parameter (ne=6000,jmax=8,nr=10000)
  real Emax,Emin,ear(0:ne),param(3),photar(ne),E,dE
  real ri(nr),kTi(nr)

! Parameters
  param(1) = 0.998    ! BH spin parameter
  param(2) = 1e4      ! rout/risco
  param(3) = 70.0     ! Inclination angle in degrees
  
! Set energy grid  
  Emax  = 5.0
  Emin  = 1e-3
  do i = 0,ne
    ear(i) = Emin * (Emax/Emin)**(real(i)/real(ne))
  end do

! Read in radius and temperature grids
  open(98,file='radial_array.csv')
  open(97,file='temperature_array_keV.csv')
  do i = 1,nr
     read(98,*)ri(i)
     read(97,*)kTi(i)
  end do
  close(98)
  close(97)
  
! Call the code to test
  call tdedisc_grid(param,ri,kTi,nr,photar)
  ! Write out model output
  do i = 1,ne
     E  = 0.5 * ( ear(i) + ear(i-1) )
     dE =         ear(i) - ear(i-1)
     write(99,*)E,E**2*photar(i)/dE
  end do
  write(99,*)"log"
  
end program wrapper

include 'amodules.f90'


subroutine setup_energy_grids()
  use internal_grids
  implicit none
  integer i
  real mybbody,bbodx(nex),dE,E
  ! Initialize
  if( firstcall )then
     firstcall = .false.
     !Define *logarithmic* internal energy grid
     Emax  = 1e2
     Emin  = 1e-4
     dloge = log10( Emax / Emin ) / real(nex)
     do i = 0,nex
       earx(i) = Emin * (Emax/Emin)**(real(i)/real(nex))
     end do
     !Blackbody with kT = 1keV, and integral of unity
     do i = 1,nex
        Emidx(i)  = 0.5 * ( earx(i) + earx(i-1) )
        dEarr(i)  = earx(i) - earx(i-1)
        ! bbodx(i)  = mybbody(1.0,Emidx(i),dEarr(i))
     end do
     !Define *logarithmic* internal energy grid
     dloge = log10( Emax / Emin ) / real(nec)
     do i = 0,nec
       earc(i) = Emin * (Emax/Emin)**(real(i)/real(nec))
     end do
     !Blackbody with kT = 1keV, and integral of unity
     do i = 1,nec
        Emidc(i)  = 0.5 * ( earc(i) + earc(i-1) )
        dEarrc(i)  = earc(i) - earc(i-1)
        ! bbodx(i)  = mybbody(1.0,Emidx(i),dEarr(i))
     end do
     !Fourier transform
     
    !  call pad4FFT(nex,bbodx,FTbbodx)
     !Assign impossible initia values to previous parameters
     aprev   = 10.d0
     mu0prev = 10.d0
  end if


end subroutine setup_energy_grids


!=======================================================================
subroutine tdedisc_grid(param,ri,kTi,nr,photar)
! Calculates observed disk spectrum
  use internal_grids
  implicit none
  integer i,j,nr,ilo,ihi, k
  real param(3),photar(nex)
  real ri(nr),kTi(nr)
  double precision a,rin,rout,inc,pi,mu0,disco,d,rnmin,rnmax,rfunc
  double precision alpha(nro,nphi),beta(nro,nphi),dOmega(nro,nphi)
  double precision mudisk
  double precision g,dlgfac,re,kT,dFe
  real E,dE,mybbody
  real dNdE_! NAN killer. 
  real dNbydE(nec),Eem,dEem
  double precision fcol,fcol0
  real kTcol
  logical needtrace
  real ri_min,ri_max,dri
  pi  = acos(-1.d0)
  
! Parameters
  a      = dble( param(1) )
  rin    = disco(a)
  rout   = dble( param(2) )! * rin
  inc    = dble( param(3) ) * pi / 180.d0
  mu0    = dble( cos(inc) )
  
  call setup_energy_grids()

! Set up radial grid
  ri_min = ri(1)
  ri_max = ri(nr)
  dri    = ( ri_max - ri_min ) / real(nr-1)
  
! Set up full GR grid
  rnmax = 300.d0                          !Sets outer boundary of full GR grid
  rnmin = rfunc(a,mu0)                    !Sets inner boundary of full GR grid
  call impactgrid(rnmin,rnmax,mu0,nro,nphi,alpha,beta,dOmega)
  d     = max( 1.0d4 , 2.0d2 * rnmax**2 ) !Sensible distance to BH  


! Do the ray tracing in full GR
  needtrace = .false.
  if( abs( a - aprev ) .gt. tiny(a) ) needtrace = .true.
  if( abs(mu0 - mu0prev) .gt. tiny(mu0) ) needtrace = .true.
  mudisk = 0.d0       !razor thin disk
  if( needtrace )then
     call dGRtrace(nro,nphi,alpha,beta,mu0,a,rin,rout,mudisk,d,pem1,re1)
  end if
  aprev = a
  mu0prev = mu0
  
! ! Loop through inner relativistic grid
  dNbydE = 0.0
  dNdE_ = 0.0
  do j = 1,nphi
     do i = 1,nro
        if( pem1(i,j) .gt. 0.d0 )then
           re = re1(i,j)
           if( re .ge. rin .and. re .le. rout )then
              !Interpolate disc temperature from input temperature grid
              ihi = ceiling( (re-ri_min)/dri ) + 1
              ihi = max( ihi , 2 )
              ilo = ihi - 1
              kT  = kTi(ilo) + ( kTi(ihi) - kTi(ilo) ) * ( re - ri(ilo) ) / dri
          
              !Calculate colour-temperature correction
              fcol0 = fcol(kT)
              !Calculate g-factor
              g = dlgfac( a,mu0,alpha(i,j),re )
              !Add to line profile
              if( kT .gt. tiny(kT) )then
                 !Calculate contribution to line profile
                 dFe = g**3 * (kT)**3 * dOmega(i,j) / fcol0
  
                 !Work out what bin this goes into
                 kTcol = fcol0 * kT
                !  n = ceiling( log10(g*kTcol) / dloge ) + nex / 2
                !  n = max( 1 , n   )
                !  n = min( n , nex )
                !  !Add to line profile
                !  diskline(n) = diskline(n) + real( dFe )              
                !  do k = 1,nex
                !   E         = 0.5 * ( earx(k) + earx(k-1) )
                !   dE        = earx(k) - earx(k-1)
                 do k = 1,nec
                  E         = 0.5 * ( earc(k) + earc(k-1) )
                  dE        = earc(k) - earc(k-1)
                  Eem        =  E / g
                  dEem       = dE / g
                  dNdE_      = g**3 * kT**4 * mybbody(kTcol,Eem,dEem) * dOmega(i,j) / dE
                  if (dNdE_ .eq. dNdE_) then 
                   dNbydE(k) = dNbydE(k) + dNdE_
                  else
                   dNbydE(k) = dNbydE(k)
                  end if 
               end do

              end if
           end if
        end if
     end do
  end do

  ! Rebin onto input grid
  call myinterp(nec,earc,dNbydE,nex,earx,photar)
  
  ! Multiply by dE
  do i = 1,nex
    E  = 0.5 * ( earx(i) + earx(i-1) )
    dE = earx(i) - earx(i-1)
    photar(i) = photar(i) * dE
 end do

  
  return
end subroutine tdedisc_grid
!=======================================================================

!=======================================================================
subroutine tdedisc_grid_for_xspec(ear,ne,param,ri,kTi,nr,ifl,photar)
! Calculates observed disk spectrum
! ifl has been left for combatability with xspec.  It can otherwise be ignored
  use internal_grids
  implicit none
  integer ne,nr,ifl,i
  real ear(0:ne),param(8),photar(ne)
  real dE,unbinned_photar(nex),px(nex),p(ne)
  real ri(nr),kTi(nr)


  ifl = 1
  
  call tdedisc_grid(param, ri,kTi,nr,unbinned_photar)

  !Now rebin onto input array
  do i = 1,nex
    px(i) = unbinned_photar(i) / dEarr(i)
  end do
  call rebinE(earx,px,nex,ear,p,ne)
  do i = 1,ne
    dE       = ear(i) - ear(i-1)
    photar(i) = p(i) * dE
  end do

  return
end subroutine tdedisc_grid_for_xspec

!-----------------------------------------------------------------------
function fcol(T)
! fcol = colour-temperature correction
! T    = true temperature in keV
  implicit none
  double precision fcol,T
  if( T .lt. 2.585d-3 )then
     fcol = 1.d0
  else if( T .lt. 8.617e-3 )then
     fcol = ( T / 2.585d-3 )**0.833
  else
     fcol = ( 72.d0 / T )**(1.d0/9.d0)
  end if
  return
end function fcol
!-----------------------------------------------------------------------





!-----------------------------------------------------------------------
      subroutine padcnv(dyn,ne,diskline,photard,conv)
! Convolves the array diskline(1:ne) with the array photard(1:ne) using
! FFTs, the result is recorded in conv(1:ne).
! This code uses extensive zero padding, and also applies a gate to
! get rid of high energy noise.
! Parameter dyn sets the dynamic range allowed in the output array.
! Anything smaller than dyn * ( the maximum value of conv ) will
! be set to zero. The value dyn = 1e-7 works very well.
      implicit none
      integer ne,i
      real diskline(ne),photard(ne),conv(ne),padconv(4*ne)
      real padline(4*ne),padphot(4*ne),photmax,dyn

! Fill padded arrays
      padline = 0.0
      padphot = 0.0
      do i = 1,ne
        padline(i+2*ne) = diskline(i)
        padphot(i+2*ne) = photard(i)
      end do

! Call the convolution code
      call FTcnv(4*ne,padline,padphot,padconv)

! Populate output array
      photmax = 0.0
      do i = 1,ne
        conv(i) = padconv(i+5*ne/2)
        photmax = max( photmax , conv(i) )
      end do

! Clean any residual edge effects
      do i = 1,ne
        if( abs(conv(i)) .lt. abs(dyn*photmax) ) conv(i) = 0.0
      end do
      
      return
      end subroutine padcnv
!-----------------------------------------------------------------------


!-----------------------------------------------------------------------
      subroutine dFTcnv(nex,line,photarx,conv)
! Takes the arrays line(1-nex/2:nex/2) and photarx(1-nex/2:nex/2)
! and convolves them to get conv(1-nex/2:nex/2)
! Uses FFTs, so nex must be a power of 2.
      implicit none
      integer nex,i
      double precision line(nex),photarx(nex),conv(nex)
      real adata(2*nex),bdata(2*nex),cdata(2*nex)
      complex ac(nex),bc(nex),cc(nex)

! Move arrays into arrays for four1
      adata = 0.0
      bdata = 0.0
      !-ve frequencies
      do i = 1,nex/2-1
        adata(2*i+nex+1) = real( line(i) )
        bdata(2*i+nex+1) = real( photarx(i) )
      end do
      !DC component
      adata(1) = real( line(nex/2) )
      bdata(1) = real( photarx(nex/2) )
      !+ve frequencies
      do i = nex/2,nex
        adata(2*i-nex+1) = real( line(i) )
        bdata(2*i-nex+1) = real( photarx(i) )
      end do
      
! Now do the inverse FFT
      call ourfour1(adata,nex,-1)
      call ourfour1(bdata,nex,-1)

! Now put into complex arrays
      do i = 1,nex
        ac(i) = complex( adata(2*i-1) , adata(2*i) )
        bc(i) = complex( bdata(2*i-1) , bdata(2*i) )
      end do

! Multiply complex numbers together
      cc = ac * bc / float(nex)

! Put back into four1 style arrays
      do i = 1,nex
        cdata(2*i-1) =  real( cc(i) )
        cdata(2*i  ) = aimag( cc(i) )
      end do
      
! Then transform back
      call ourfour1(cdata,nex,1)

! Move arrays back into original format
      !-ve frequencies
      do i = 1,nex/2-1
        conv(i) = dble( cdata(2*i+nex+1) )
      end do
      !DC component
      conv(nex/2) = dble( cdata(1) )
      !+ve frequencies
      do i = nex/2,nex
        conv(i) = dble( cdata(2*i-nex+1) )
      end do
      
      return
      end
!-----------------------------------------------------------------------


!-----------------------------------------------------------------------
      subroutine FTcnv(nex,line,photarx,conv)
! Takes the arrays line(1-nex/2:nex/2) and photarx(1-nex/2:nex/2)
! and convolves them to get conv(1-nex/2:nex/2)
! Uses FFTs, so nex must be a power of 2.
      implicit none
      integer nex,i
      real line(nex),photarx(nex),conv(nex)
      real adata(2*nex),bdata(2*nex),cdata(2*nex)
      complex ac(nex),bc(nex),cc(nex)

! Move arrays into arrays for four1
      adata = 0.0
      bdata = 0.0
      !-ve frequencies
      do i = 1,nex/2-1
        adata(2*i+nex+1) = line(i)
        bdata(2*i+nex+1) = photarx(i)
      end do
      !DC component
      adata(1) = line(nex/2)
      bdata(1) = photarx(nex/2)
      !+ve frequencies
      do i = nex/2,nex
        adata(2*i-nex+1) = line(i)
        bdata(2*i-nex+1) = photarx(i)
      end do
      
! Now do the inverse FFT
      call ourfour1(adata,nex,-1)
      call ourfour1(bdata,nex,-1)
      
! Now put into complex arrays
      do i = 1,nex
        ac(i) = complex( adata(2*i-1) , adata(2*i) )
        bc(i) = complex( bdata(2*i-1) , bdata(2*i) )
      end do

! Multiply complex numbers together
      cc = ac * bc / float(nex)

! Put back into four1 style arrays
      do i = 1,nex
        cdata(2*i-1) =  real( cc(i) )
        cdata(2*i  ) = aimag( cc(i) )
      end do
      
! Then transform back
      call ourfour1(cdata,nex,1)
      
! Move arrays back into original format
      !-ve frequencies
      do i = 1,nex/2-1
        conv(i) = cdata(2*i+nex+1)
      end do
      !DC component
      conv(nex/2) = cdata(1)
      !+ve frequencies
      do i = nex/2,nex
        conv(i) = cdata(2*i-nex+1)
      end do
      
      return
      end
!-----------------------------------------------------------------------


!-----------------------------------------------------------------------
      SUBROUTINE ourfour1(data,nn,isign)
      INTEGER isign,nn
      REAL data(2*nn)
      INTEGER i,istep,j,m,mmax,n
      REAL tempi,tempr
      DOUBLE PRECISION theta,wi,wpi,wpr,wr,wtemp
      n=2*nn
      j=1
      do 11 i=1,n,2
        if(j.gt.i)then
          tempr=data(j)
          tempi=data(j+1)
          data(j)=data(i)
          data(j+1)=data(i+1)
          data(i)=tempr
          data(i+1)=tempi
        endif
        m=n/2
1       if ((m.ge.2).and.(j.gt.m)) then
          j=j-m
          m=m/2
        goto 1
        endif
        j=j+m
11    continue
      mmax=2
2     if (n.gt.mmax) then
        istep=2*mmax
        theta=6.28318530717959d0/(isign*mmax)
        wpr=-2.d0*sin(0.5d0*theta)**2
        wpi=sin(theta)
        wr=1.d0
        wi=0.d0
        do 13 m=1,mmax,2
          do 12 i=m,n,istep
            j=i+mmax
            tempr=sngl(wr)*data(j)-sngl(wi)*data(j+1)
            tempi=sngl(wr)*data(j+1)+sngl(wi)*data(j)
            data(j)=data(i)-tempr
            data(j+1)=data(i+1)-tempi
            data(i)=data(i)+tempr
            data(i+1)=data(i+1)+tempi
12        continue
          wtemp=wr
          wr=wr*wpr-wi*wpi+wr
          wi=wi*wpr+wtemp*wpi+wi
13      continue
        mmax=istep
      goto 2
      endif
      return
      end
!-----------------------------------------------------------------------

      

!-----------------------------------------------------------------------
function mybbody(kT,E,dE)
! Blackbody function in terms of number of photons with energy
! between E-dE/2 and E+dE/2. i.e. This is photar!
! Function is normalized such that the integrated energy flux is 1
! i.e. sum E * photar(E) = 1
  implicit none
  real mybbody,E,dE,kT
  real pi,fac,f
  pi   = acos(-1.0)
  fac  = E/kT
  if(fac .lt. 1e-3)then
    f = E * kT   !Using a Taylor expansion
  else
    if (fac .lt. 70.) then
      f = E**2 / ( exp(fac) - 1.0 ) 
    else
      f = E**2 * exp(-fac)
    end  if
  end if
  mybbody = f * (15.0/pi**4) / kT**4 * dE
  return
end function mybbody
!-----------------------------------------------------------------------


!-----------------------------------------------------------------------
function dlgfac(a,mu0,alpha,r)
!c Calculates g-factor for a disk in the BH equatorial plane
  implicit none
  double precision dlgfac,a,mu0,alpha,r
  double precision sin0,omega,Delta,Sigma2,gtt,gtp,gpp
  sin0   = sqrt( 1.0 - mu0**2 )
  omega  = 1. / (r**1.5+a)
  Delta  = r**2 - 2*r + a**2
  Sigma2 = (r**2+a**2)**2 - a**2 * Delta
  gtt    = 4*a**2/Sigma2 - r**2*Delta/Sigma2
  gtp    = -2*a/r
  gpp    = Sigma2/r**2
  dlgfac = sqrt( -gtt - 2*omega*gtp - omega**2.*gpp )
  dlgfac = dlgfac / ( 1.+omega*alpha*sin0 )
  return
end function dlgfac
!-----------------------------------------------------------------------


!-----------------------------------------------------------------------
subroutine dGRtrace(nro,nphi,alpha,beta,mu0,spin,rmin,rout,mudisk,d,pem1,re1)
! Traces rays in the Kerr metric for a camera defined by the impact
! parameters at infinity: alpha(nro,nphi) and beta(nro,nphi).
! Traces back to a disk defined by mudisk = cos(theta_disk), where
! theta_disk is the angle between the vertical and the disk surface.
! i.e. tan( theta_disk ) = 1 / (h/r)
! OUTPUT:
! pem1(nro,nphi)
! pem > 1: there is a solution
! pem = -1 photon goes to infinity without hitting disk surface
! pem = -2 photon falls into horizon without hitting disk surface
! re1(nro,nphi)      radius that the geodesic hits the disc
  use blcoordinate     ! This is a YNOGK module
  implicit none
  integer nro,nphi,i,j
  double precision alpha(nro,nphi),beta(nro,nphi),mu0,spin,rmin,rout,mudisk,d
  double precision pem1(nro,nphi),re1(nro,nphi)
  double precision cos0,sin0,scal,velocity(3),f1234(4),lambda,q
  double precision pem,re,mucros,phie,taudo,sigmacros      
  cos0  = mu0
  sin0  = sqrt(1.0-cos0**2)
  scal     = 1.d0
  velocity = 0.d0
  re1      = 0.0
  do i = 1,nro
    do j = 1,NPHI
      call lambdaq(-alpha(i,j),-beta(i,j),d,sin0,cos0,spin,scal,velocity,f1234,lambda,q)
      pem = Pemdisk(f1234,lambda,q,sin0,cos0,spin,d,scal,mudisk,rout,rmin)  !Can try rin instead of rmin to save an if statement
      pem1(i,j) = pem
      !pem > 1 means there is a solution
      !pem < 1 means there is no solution
      if( pem .gt. 0.0d0 )then
        call YNOGK(pem,f1234,lambda,q,sin0,cos0,spin,d,scal,re,mucros,phie,taudo,sigmacros)
        re1(i,j)    = re
      end if
    end do
  end do
  return
end subroutine dGRtrace
!-----------------------------------------------------------------------



!-----------------------------------------------------------------------
subroutine impactgrid(rnmin,rnmax,mu0,nro,nphi,alpha,beta,dOmega)
! Calculates a grid of impact parameters
! INPUT:
! rnmin        Sets inner edge of impact parameter grid
! rnmax        Sets outer edge of impact parameter grid
! mu0          Sets `eccentricity' of the grid
! nro          Number of steps in radial impact parameter (b)
! nphi         Number of steps in azimuthal impact parameter (phi)
! OUTPUT:
! alpha(nro,nphi)   Horizontal impact parameter
! beta(nro,nphi)    Vertical impact parameter
! dOmega(nro,nphi)  dalpha*dbeta
  implicit none
  integer nro,nphi,i,j
  double precision rnmin,rnmax,mu0,alpha(nro,nphi),beta(nro,nphi)
  double precision dOmega(nro,nphi),mueff,pi,rar(0:nro),dlogr,rn(nro)
  double precision logr,phin
  pi     = acos(-1.d0)

  mueff = max( mu0 , 0.3d0 )
  
  rar(0) = rnmin
  dlogr  = log10( rnmax/rnmin ) / dble(nro)
  do i = 1,NRO
    logr = log10(rnmin) + dble(i) * dlogr
    rar(i)    = 10.d0**logr
    rn(i)     = 0.5 * ( rar(i) + rar(i-1) )
    do j = 1,nphi
       domega(i,j) = rn(i) * ( rar(i) - rar(i-1) ) * mueff * 2.d0 * pi / dble(nphi)
       phin       = (j-0.5) * 2.d0 * pi / dble(nphi) 
       alpha(i,j) = rn(i)  * sin(phin)
       beta(i,j)  = rn(i) * cos(phin) * mueff
    end do
  end do
  
  return
end subroutine impactgrid
!-----------------------------------------------------------------------


!-----------------------------------------------------------------------
function dISCO(a)
  !ISCO in Rg 
  implicit none
  double precision a,dISCO,z1,z2
  if(a.ge.1.0)then
      z1 = (a**2.0 - 1.0)**(1.0/3.0)
      z1 = z1 * ( (1.0+a)**(1.0/3.0)-(a-1.0)**(1.0/3.0))+1.0
  else
      z1 = ( 1.0 - a**2.0 )**(1.0/3.0)
      z1 = z1 * ( (1.0+a)**(1.0/3.0)+(1.0-a)**(1.0/3.0))+1.0
  end if 
      
  z2 = sqrt( 3.0 * a**2.0 + z1**2.0 )
  if(a.ge.0.0)then
    dISCO = 3.0 + z2 - sqrt( (3.0-z1) * (3.0 + z1 + 2.0*z2) )
  else
    dISCO = 3.0 + z2 + sqrt( (3.0-z1) * (3.0 + z1 + 2.0*z2) )
  end if
  return
end function dISCO
!-----------------------------------------------------------------------

  

!-----------------------------------------------------------------------
function rfunc(a,mu0)
! Sets minimum rn to use for impact parameter grid depending on mu0
! This is just an analytic function based on empirical calculations:
! I simply set a=0.998, went through the full range of mu0, and then
! calculated the lowest rn value for which there was a disk crossing.
! The function used here makes sure the calculated rnmin is always
! slightly lower than the one required.
  implicit none
  double precision rfunc,mu0,a
  if( a .gt. 0.8 )then
    rfunc = 1.5d0 + 0.5d0 * mu0**5.5d0
    rfunc = min( rfunc , -0.1d0 + 5.6d0*mu0 )
    rfunc = max( 0.1d0 , rfunc )
  else
    rfunc = 3.0d0 + 0.5d0 * mu0**5.5d0
    rfunc = min( rfunc , -0.2d0 + 10.0d0*mu0 )
    rfunc = max( 0.1d0 , rfunc )
  end if
  end function rfunc
!-----------------------------------------------------------------------


!-----------------------------------------------------------------------
subroutine getrgrid(rnmin,rnmax,mueff,nro,nphi,rn,domega)
! Calculates an r-grid that will be used to define impact parameters
  implicit none
  integer nro,nphi,i
  double precision rnmin,rnmax,mueff,rn(nro),domega(nro)
  double precision rar(0:nro),dlogr,logr,pi
  pi     = acos(-1.d0)
  rar(0) = rnmin
  dlogr  = log10( rnmax/rnmin ) / dble(nro)
  do i = 1,NRO
    logr = log10(rnmin) + dble(i) * dlogr
    rar(i)    = 10.d0**logr
    rn(i)     = 0.5 * ( rar(i) + rar(i-1) )
    domega(i) = rn(i) * ( rar(i) - rar(i-1) ) * mueff * 2.d0 * pi / dble(nphi)
  end do
  return
end subroutine getrgrid
!-----------------------------------------------------------------------


!-----------------------------------------------------------------------
      subroutine drandphithick(alpha,beta,cosi,costheta,r,phi)
!
! A disk with an arbitrary thickness
! The angle between the normal to the midplane and the disk surface is theta
! The inclination angle is i
      implicit none
      double precision alpha,beta,cosi,sini,r,phi
      double precision pi,costheta,sintheta,x,a,b,c,det
      double precision mu,sinphi
!      double precision muplus,muminus,ra,rb,rab,xplus1,xminus1,xplus2,xminus2
      pi = acos(-1.d0)
      sintheta = sqrt( 1.d0 - costheta**2 )
      sini     = sqrt( 1.d0 - cosi**2 )
      x        = alpha / beta
      if( abs(alpha) .lt. abs(tiny(alpha)) .and. abs(beta) .lt. abs(tiny(beta))  )then
        mu = 0.d0
        r  = 0.d0
      else if( abs(beta) .lt. abs(tiny(beta)) )then
        mu     = sini*costheta/(cosi*sintheta)
        sinphi = sign( 1.d0 , alpha ) * sqrt( 1.d0 - mu**2 )
        r      = alpha / ( sintheta * sinphi )
      else if( abs(alpha) .lt. abs(tiny(alpha)) )then
        mu     = 1.d0
        sinphi = 0.d0
        r      = beta / ( sini*costheta - cosi*sintheta )
      else
        a      = sintheta**2 + x**2*cosi**2*sintheta**2
        b      = -2*x**2*sini*cosi*sintheta*costheta
        c      = x**2*sini**2*costheta**2-sintheta**2
        det    = b**2 - 4.d0 * a * c
        if( det .lt. 0.d0 ) write(*,*)"determinant <0!!!"
        if( beta .gt. 0.d0 )then
          mu     = ( -b + sqrt( det ) ) / ( 2.d0 * a )
        else
          mu     = ( -b - sqrt( det ) ) / ( 2.d0 * a )
        end if
        sinphi = sign( 1.d0 , alpha ) * sqrt( 1.d0 - mu**2 )
        r      = alpha / ( sintheta * sinphi )
      end if
      phi = atan2( sinphi , mu )
      return
      end subroutine drandphithick
!-----------------------------------------------------------------------


!-----------------------------------------------------------------------
      subroutine rebinE(earx,px,nex,ear,p,ne)
      !General rebinning scheme, should be nice and robust - BUT IT FUCKING ISN'T
      !i,nex,earx,px = input
      !j,ne,ear,p    = output
      implicit none
      integer i,nex,j,ne
      real earx(0:nex),ear(0:ne),px(nex),p(ne),Ehigh,Elow,upper,lower
      real FRAC,Ej,Ei,pi,Ei2,pi2,grad,cons
      logical interp
      i = 1
      do j = 1,ne
        p(j) = 0.0
        Ehigh = ear(j)
        Elow  = ear(j-1)
        do while( earx(i) .le. Elow .and. i .lt. nex )
          i = i + 1
        end do
        interp = .true.
        do while(earx(i-1).lt.Ehigh.and.i.lt.nex)
          lower = MAX( earx(i-1) , Elow  )
          upper = MIN( earx(i)   , Ehigh )
          FRAC  = (upper-lower) / ( Ehigh - Elow )
          p(j)  = p(j) + px(i) * FRAC
          i = i + 1
          interp = .false.
        end do
        if( interp )then
          !Work out if it's ok to interpolate
          if( Elow  .ge. earx(nex) ) interp = .false.
          if( Ehigh .le. earx(0)   ) interp = .false.
          if( i     .ge. nex-1     ) interp = .false.
        end if
        if( interp )then
          !write(*,*)"need to interpolate!"
          !p(j) is interpolation between px(i) and px(i+1)
          !unless i=nex, in which case p(j) = px(i)
          Ej = 0.5 * ( Ehigh + Elow )        !Centre of newbin
          Ei = 0.5 * ( earx(i+1) + earx(i) ) !Centre of one oldbin
          pi = px(i+1)         !Value at bin centre
          if( Ei .eq. Ej )then
             p(j) = pi
          else
            if( Ei .gt. Ej )then
              Ei2 = 0.5 * (earx(i) + earx(i-1) )
              pi2 = px(i)
            else
              Ei2 = 0.5 * (earx(i+2) + earx(i+1) )
              pi2 = px(+2)
            end if
            grad = ( pi - pi2 ) / ( Ei - Ei2 )
            cons = 0.5 * ( pi + pi2 - grad*(Ei+Ei2) )
            p(j) = grad * Ej + cons
          end if
        end if
        if( i .gt. 2 ) i = i - 1
      end do
      RETURN
      END
!-----------------------------------------------------------------------



!-----------------------------------------------------------------------
subroutine pad4FFT(ne,photar,padFT)
! Takes spectrum photar(1:ne), pads out with zeros to make it a length
! of 4*ne, and Fourier transforms to padFT(1:4*ne), which is a function
! of 1/E
  implicit none
  integer ne,i
  real photar(ne),padphot(4*ne)
  complex padFT(4*ne)

! Pad out the array
  padphot = 0.0
  do i = 1,ne
     padphot(i+2*ne) = photar(i)
  end do

! Call the energy Fourier transform code
  call E_FT(4*ne,padphot,padFT)

  return
end subroutine pad4FFT      
!-----------------------------------------------------------------------




!-----------------------------------------------------------------------
subroutine pad4invFFT(dyn,ne,padFT,conv)
! Takes padFT(1:4*ne), and zero-padded function of 1/E and inverse
! Fourier transforms to conv(ne), which is a non-zero padded spectrum
  implicit none
  integer ne,i
  real dyn,conv(ne),padconv(4*ne),photmax
  complex padFT(4*ne)

! Inverse Fourier transform padded FT
  call E_invFT(4*ne,padFT,padconv)

! Populate output array
  photmax = 0.0
  do i = 1,ne
    conv(i) = padconv(i+5*ne/2)
    photmax = max( photmax , conv(i) )
  end do

! Clean any residual edge effects
  do i = 1,ne
    if( abs(conv(i)) .lt. abs(dyn*photmax) ) conv(i) = 0.0
  end do

  return
end subroutine pad4invFFT
!-----------------------------------------------------------------------





!-----------------------------------------------------------------------
subroutine E_invFT(nex,cc,conv)
! Takes the complex array cc(1:nex), which is a function of 1/E
! and inverse Fourier transforms to get back a real spectrum as a
! function of E, conv(1:nex)
  implicit none
  integer nex,i
  real conv(nex),cdata(2*nex)
  complex cc(nex)

! Put back into four1 style arrays
  do i = 1,nex
    cdata(2*i-1) =  real( cc(i) )
    cdata(2*i  ) = aimag( cc(i) )
  end do
      
! Then transform back
  call ourfour1(cdata,nex,1)
      
! Move arrays back into original format
  !-ve frequencies
  do i = 1,nex/2-1
    conv(i) = cdata(2*i+nex+1)
  end do
  !DC component
  conv(nex/2) = cdata(1)
  !+ve frequencies
  do i = nex/2,nex
    conv(i) = cdata(2*i-nex+1)
  end do
  return
end subroutine E_invFT
!-----------------------------------------------------------------------



!-----------------------------------------------------------------------
subroutine E_FT(nex,photarx,bc)
! Takes the real array photarx(1:nex), which is a spectrum as a
! function of photon energy E and Fourier transforms to bc(1:nex),
! which is complex and a function of 1/E.
! Uses FFTs, so nex must be a power of 2.
! Uses the inverse transform of four1.
  implicit none
  integer nex,i
  real photarx(nex)
  real bdata(2*nex)
  complex bc(nex)

! Move arrays into arrays for four1
  bdata = 0.0
  !-ve frequencies
  do i = 1,nex/2-1
    bdata(2*i+nex+1) = photarx(i)
  end do
  !DC component
  bdata(1) = photarx(nex/2)
  !+ve frequencies
  do i = nex/2,nex
    bdata(2*i-nex+1) = photarx(i)
  end do
      
! Now do the inverse FFT
  call ourfour1(bdata,nex,-1)
      
! Now put into complex arrays
  do i = 1,nex
    bc(i) = complex( bdata(2*i-1) , bdata(2*i) ) / sqrt(float(nex))
  end do

  return
end subroutine E_FT
!-----------------------------------------------------------------------


!-----------------------------------------------------------------------
subroutine myinterp(nfx,farx,Gfx,nf,far,Gf)
  ! Interpolates the function Gfx from the grid farx(0:nfx) to the
  ! function Gf on the grid far(0:nf)
    implicit none
    integer nfx,nf
    real farx(0:nfx),Gfx(nfx),far(0:nf),Gf(nf)
    integer ix,j
    real fx(nfx),f,fxhi,Gxhi,fxlo,Gxlo
  ! Define grid of central input frequencies
    do ix = 1,nfx
        fx(ix) = 0.5 * ( farx(ix) + farx(ix-1) )
    end do
  ! Run through grid of central output frequencies
    ix = 1
    do j = 1,nf
        !Find the input grid frequencies either side of the current
        !output grid frequency
        f = 0.5 * ( far(j) + far(j-1) )
        do while( fx(ix) .lt. f .and. ix .lt. nfx )
          ix = ix + 1
        end do
        ix = max( 2 , ix )
        fxhi = fx(ix)
        Gxhi = Gfx(ix)
        ix = ix - 1
        fxlo = fx(ix)
        Gxlo = Gfx(ix)
        !Interpolate
        Gf(j) = Gxlo + ( Gxhi - Gxlo ) * ( f - fxlo ) / ( fxhi - fxlo )
    end do
    return
  end subroutine myinterp
  !-----------------------------------------------------------------------
    
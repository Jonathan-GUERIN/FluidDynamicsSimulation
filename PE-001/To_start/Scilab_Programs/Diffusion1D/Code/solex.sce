function y=solex(tau,eta,nmax)
  
  sumtheta=0;
  for n=0:nmax
    nn=2*n+1;
    sumtheta=sumtheta + (1/nn)*exp(-tau*(nn*%pi/2)**2)*sin(nn*%pi*eta/2)
  end
  
  y=(4/%pi)*sumtheta
  
endfunction

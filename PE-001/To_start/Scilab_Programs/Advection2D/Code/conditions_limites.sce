
// condition d'entrée

   for i=1:imax
      rho(i,1)=rho_inlet(x(i));
   end
   
   for j=1:jmax
   	rho(1,j)=rho_inlet(x(1));
   end
   
// condition de sortie

	  for i=1:imax
	  	rho(i,jmax)=rho(i,jmax-1);
	  end
	  
	  for j=1:jmax
	  	rho(imax,j)=rho(imax-1,j);
	  end
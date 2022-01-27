// theta : Tableau de stockage de la temperature adimensionnee
// xi : coordonnee spatiale adimensionnee
// dxi : Pas d'espace adimensionne 
// Fo: nb de Fourier = temps adimensionne ecoule depuis l'instant initial
// dFo : pas de temps adimensionne
// freq_ech : frequence d'affichage (en iterations) de la solution

  clearglobal();
  clear();
  clf(0);clf(1);

  exec('param.sce'); 	// chargement des parametres  
  
  exec('init.sci'); // chargement de la fonction "condition initiale"

// definition du maillage 

   dx = (x2-x1) / (imax-1);
   
   x=linspace(x1,x2,imax);
   x=x';
   
   temps   = 0.0;
  
// trace distribution initiale
  
  rho=init(x);
  
  xset("window",0)
 	plot2d(x,rho,1)
 	  	  	
// BOUCLE PRINCIPALE
    
   for iter = 1:itermax   
    
     dt = CFL*dx/abs(a);
     
   temps=temps+dt;
      
      if kscheme==0 then
        flux_num(2:imax)=0.5*a*(rho(1:imax-1)+rho(2:imax));
      elseif kscheme==1 then
        flux_num(2:imax)=a*rho(1:imax-1);
      elseif kscheme==2 then
        flux_num(2:imax)=a*rho(2:imax);
      elseif kscheme==3 then
        flux_num(2:imax)=0.5*a*(rho(1:imax-1)+rho(2:imax))-0.5*abs(a)*(rho(2:imax)-rho(1:imax-1));  
      end  
      rho(2:imax-1)=rho(2:imax-1)-(dt/dx)*(flux_num(3:imax)-flux_num(2:imax-1));
      
          // conditions aux limites
    rho(1)=rho(imax-1);
    rho(imax)=rho(2);
     
     //    rhoex=init(x-a*temps);
     rhoex=init(x);
  
// Sortie graphique
if modulo(iter,freq_dist) == 0 then
  xset("window",1)
   plot2d(x,[rho,rhoex],[1,2])
   end
      
   end
   
  

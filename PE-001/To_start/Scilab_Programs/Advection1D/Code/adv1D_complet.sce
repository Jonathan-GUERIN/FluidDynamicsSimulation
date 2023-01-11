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
   for i=1:imax
      x(i)=x1+(i-1)*dx;
    end

   temps   = 0.0;
  
// trace distribution initiale
  
  for i=1:imax
  rho(i)=init(x(i));
  end
  
  xset("window",0)
 	plot2d(x,rho,1)
 	  	  	
// BOUCLE PRINCIPALE
    
   for iter = 1:itermax
     
     dt=CFL*dx/max(a*sin(2*%pi*temps),0.1);
     
     temps=temps+dt;
   
    for i=2:imax
     // flux_num(i)=a*0.5*(rho(i-1)+rho(i))      
     // flux_num(i)=a*rho(i-1);
     flux_num(i)=0.5*a*sin(2*%pi*temps)*(rho(i-1)+rho(i))-0.5*abs(a*sin(2*%pi*temps))*(rho(i)-rho(i-1));
    end
     
    for i=2:imax-1
      rho(i)=rho(i)-(dt/dx)*(flux_num(i+1)-flux_num(i));
    end
    
    // conditions aux limites
    rho(1)=rho(imax-1);
    rho(imax)=rho(2);
    
    for i=1:imax
    rhoex(i)=init(x(i)-a*temps);
    end
    
// Sortie graphique
if modulo(iter,freq_dist) == 0 then
  xset("window",0)
//   plot2d(x,rhoex,2)
   plot2d(x,[rho,rhoex],[1,2])
   end
      
   end
   
  

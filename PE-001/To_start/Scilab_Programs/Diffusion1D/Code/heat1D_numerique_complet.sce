// theta : Tableau de stockage de la temperature adimensionnee
// xi : coordonnee spatiale adimensionnee
// dxi : Pas d'espace adimensionne 
// Fo: nb de Fourier = temps adimensionne ecoule depuis l'instant initial
// dFo : pas de temps adimensionne
// freq_ech : frequence d'affichage (en iterations) de la solution

  clearglobal();
  clear();
  clf(0);clf(1);clf(2);clf(3);

  exec('param.sce'); 	// chargement des parametres  
  
  exec('solex.sce'); // chargement de la fonction "solution exacte"
  
  exec('diffus.sce'); // chargement de la fonction diffusivite thermique (adimensionnee)

// definition du maillage 

   dxi = 2. / (imax-1);
   for i=1:imax
      xi(i)=(i-1)*dxi;
    end

   Fo   = 0.0;
  
// trace distribution initiale
    
  for i=2:imax-1
  theta(i)=1;
  end
  theta(1)=0;
  theta(imax)=0;
  
  xset("window",0)
 	plot2d(xi,theta,-5)
 	  	  	
// BOUCLE PRINCIPALE
 	  	  	
   timer();
 	  	  	
   for iter = 1:itermax
   
   phimax=0;
   for i=2:imax-1
   phimax=max(phimax,phi(theta(i)));
   end
   
   dFo=acc*0.5*dxi*dxi/phimax;
   if iter==1 then
   dFoinit=dFo;
   end
   
   Fo=Fo+dFo;
   
    for i=2:imax
      flux_num(i)=phi(0.5*(theta(i-1)+theta(i)))*(theta(i)-theta(i-1))/dxi;
    end
     
    for i=2:imax-1
      theta(i)=theta(i)+(dFo/dxi)*(flux_num(i+1)-flux_num(i));
    end
    
    for i=1:imax
    thetaex(i)=solex(Fo,xi(i),nmode);
    end
    
// Sortie graphique
   if modulo(iter,freq_dist) == 0 then
   xset("window",0)
   plot2d(xi,[theta,thetaex],[-5,2])
   xset("window",2)
   plot2d(iter,dFo/dFoinit,-5)
   xset("window",3)
   plot2d(L*L*Fo/alpha,[T1+(Ti-T1)*theta((imax+1)/2),T1+(Ti-T1)*thetaex((imax+1)/2)],[-5,-2])
   end
      
   end
   
   tfin_dim = L*L*Fo/alpha;
   disp(tfin_dim,'temps final (s) = ');
   Tex=T1+(Ti-T1)*thetaex;
   T=T1+(Ti-T1)*theta;
   x=L*(xi-1);
   xset("window",1)
   plot2d(x,[T,Tex],[-5,2]);
   
   tps_CPU=timer();
   disp(tps_CPU,'Temps CPU (s) = ');

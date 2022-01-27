// theta : Tableau de stockage de la temperature adimensionnee
// xi : coordonnee spatiale adimensionnee
// dxi : Pas d'espace adimensionne 
// Fo: nb de Fourier = temps adimensionne ecoule depuis l'instant initial
// dFo : pas de temps adimensionne
// freq_ech : frequence d'affichage (en iterations) de la solution

  clearglobal();
  clear();
  clf();

  exec('param.sce'); 	// chargement des parametres  
  exec('solex.sce'); // chargement de la fonction "solution exacte"
  exec('diffus.sce'); // chargement de la fonction diffusivite thermique (adimensionnee)

// definition du maillage 
   dxi = 2. / (imax-1);
   xi=linspace(0,2,imax);
   xi=xi';

   Fo   = 0.0;
  
// trace distribution initiale

  theta(1:imax)=1;
  theta(1)=0;
  theta(imax)=0;
  
  subplot(221)
  xtitle("Distribution initiale pour theta","xi","theta")
  plot2d(xi,theta,-5)
 	  	  	
// BOUCLE PRINCIPALE
 	  	  	
   timer();
 	  	  	
   for iter = 1:itermax
   
   phimax=max(phi(theta));
   
   dFo=acc*0.5*dxi*dxi/phimax;
   if iter==1 then
   dFoinit=dFo;
   end
   
   Fo=Fo+dFo;
   
   flux_num(2:imax)=phi(0.5*(theta(1:imax-1)+theta(2:imax))).*(theta(2:imax)-theta(1:imax-1))/dxi;
  
   theta(2:imax-1)=theta(2:imax-1)+(dFo/dxi)*(flux_num(3:imax)-flux_num(2:imax-1));
    
   thetaex=solex(Fo,xi,nmode);
    
// Sortie graphique
   if modulo(iter,freq_dist) == 0 then
   subplot(221)
   xtitle("Evolution theta et thetaex (lineaire) au cours du temps","xi","theta")
   plot2d(xi,[theta,thetaex],[-5,2])
   subplot(222)
   xtitle("Evolution du pas de temps dFo en fonction des iterations","Iterations","dFo")
   plot2d(iter,dFo/dFoinit,-5)
   subplot(223)
   xtitle("Evolution de T(x-0) et Tex(x=0) (lineaire) au cours du temps","Temps (s)","Temperature (¡C)")
   plot2d(L*L*Fo/alpha,[T1+(Ti-T1)*theta((imax+1)/2),T1+(Ti-T1)*thetaex((imax+1)/2)],[-5,-2])
   end
      
   end
   
   tfin_dim = L*L*Fo/alpha;
   disp(tfin_dim,'temps final (s) = ');
   Tex=T1+(Ti-T1)*thetaex;
   T=T1+(Ti-T1)*theta;
   x=L*(xi-1);
   subplot(224)
   xtitle("Distribution finale de T et Tex (lineaire)","x (m)","Temperature (¡C)")
   plot2d(x,[T,Tex],[-5,2]);
   
   tps_CPU=timer();
   disp(tps_CPU,'Temps CPU (s) = ');
    

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
 	plot2d(xi,theta,1)
 	  	  	
// BOUCLE PRINCIPALE
    
   for iter = 1:itermax
   
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
   plot2d(xi,[theta,thetaex],[1,2])
   end
      
   end
   
   tfin_dim = L*L*Fo/alpha;
   disp(tfin_dim,'temps final (s) = ');
   Tex=T1+(Ti-T1)*thetaex;
   T=T1+(Ti-T1)*theta;
   x=L*(xi-1);
   xset("window",1)
   plot2d(x,[T,Tex],[1,2]);
   
  

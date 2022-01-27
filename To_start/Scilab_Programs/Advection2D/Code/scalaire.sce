// variables
//
// rho : Tableau de stockage de la variable advectee
// dt : Pas de temps
// dx,dy : Pas d'espace
// fnum1, fnum2 : flux numeriques
// fphy1, fphy2 : flux physiques
// x,y : coordonnée spatiale
// freq_ech : fréquence d'affichage de la solution

	 clear();
   clearglobal();
   clf(0);clf(1);clf(2);clf(3);clf(4);
	
   exec('param.sce'); 	// chargement des paramètres de simulation 
      
   exec("model.sci");   // chargement des fonctions
      
   dx = l_dom_x / (imax-1);
   dy = l_dom_y / (jmax-1);

   disp(imax,'Nb cellules suivant x : '); 
   disp(jmax,'Nb cellules suivant y : '); 
   disp(xmin,'Abcisse minimale : ');
   disp(ymin,'Ordonnée minimale : '); 
   disp(xmin+l_dom_x,'Abcisse maximale : '); 
   disp(ymin+l_dom_y,'Ordonnée maximale : '); 
   disp(dx,'Pas d''espace suivant x : ');
   disp(dy,'Pas d''espace suivant y : ');
   disp(cfl,'Nombre CFL : ');
   disp(freq_ech,'Frequence des sorties : '); 
   
   for i=1:imax
      x(i)=xmin+(i-1)*dx;
   end
   for j=1:jmax
      y(j)=ymin+(j-1)*dy;
   end
   
   for i=1:imax
   for j=1:jmax
      rho(i,j)=0;
   end	
   end
   
   	for i=1:imax
	  for j=1:jmax
	  rhoex(i,j)=rho_ex(x(i),y(j));
	  end
	  end
 	  
   disp('la simulation demarre...');   
   
   exec('conditions_limites.sce');
   
// BOUCLE PRINCIPALE

	 norm_res=1;
	 iter=0;
    
   while log10(norm_res) > -5
   
	    exec('pas_de_temps.sce');
	    exec('flux_numerique.sce');
	    
      rhoold=rho;
      
      for i=2:imax-1
      for j=2:jmax-1
      rho(i,j) = rho(i,j)-(dt/dx)*(f1num(i,j) - f1num(i-1,j))-(dt/dy)*(f2num(i,j) - f2num(i,j-1));
      end
      end
      
      exec('conditions_limites.sce');
      
      iter=iter+1;
      
      norm_res=0;
      for i=1:imax
      for j=1:jmax
      norm_res=norm_res+(rho(i,j)-rhoold(i,j))**2;
      end
      end
      norm_res=sqrt(norm_res/(imax*jmax));
      
      norm_err=0;
      for i=1:imax
      for j=1:jmax
      norm_err=norm_err+(rho(i,j)-rhoex(i,j))**2;
      end
      end
      norm_err=sqrt(norm_err/(imax*jmax));
     
      // Sortie graphique
      if modulo(iter,freq_ech) == 0 then
      	xset("window",0)
 	  		plot2d(iter,[log10(norm_res),log10(norm_err)],[-1,-2])
 	  	end
 	   
   end
   
   disp('solution stationnaire atteinte');     
	 disp(log10(dx),'log10(h) = ');
	 disp(log10(norm_err),'log10(norm_err) = ');
	 
	 xset("window",1)
	 plot3d(x,y,rho);	
	 
   xset("window",2)
	 contour2d(x,y,rho,20);
	   
	  xset("window",3)
	  contour2d(x,y,rhoex,20);
	  
	  xset ("window",4)
	  plot2d(x(1:imax),[rho(1:imax,jmax),rhoex(1:imax,jmax)],[1,2]);


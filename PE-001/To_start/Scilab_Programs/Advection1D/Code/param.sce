// parametres physiques du probleme
//  vitesse d'advection a  (m/s)
//  dimension du domaine = segment [x1,x2]

a=0.1;
x1=-5;;
x2=5;;
		
// parametres numeriques de la resolution
//
// imax : Nombre de cellules de calcul
// acc : facteur d'acceleration du pas de temps
// itermax : Nombre d'iterations  en temps a effectuer
// freq_dist : frequence des sorties de distributions spatiales
//

  imax=201;

  acc=1.;

  itermax=400;
  
  freq_dist=400;
  
// kscheme =0 -> centr� simple, 1 -> d�centr� � gauche, 2 -> d�centr� � droite, 3 -> d�centr� g�n�ral
  kscheme=3;
  

// choix du pas d'avancement en temps

CFL=0.5;

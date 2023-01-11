// parametres physiques du probleme
//	* temperature initiale du mur Ti (en °C)
//	* temperature imposée au niveau des parois gauche et droite : T1 (en °C)
//  * L = 1/2-epaisseur du mur
//	* coefficient de diffusivite thermique : alpha (en m2/s) 
//    la valeur fournie correspond a de l'acier
//
		Ti=100;
		T1=20;
		L=0.1;
		alpha=0.4E-5;
		
		T0=20;
		r=0.5;
		
// parametres numeriques de la resolution
//
// imax : Nombre de cellules de calcul
// acc : facteur d'acceleration du pas de temps adimensionne
// kimpli : integration en temps explicite (0) ou implicite (1)
// itermax : Nombre d'iterations  en temps a effectuer
// freq_dist : frequence des sorties de distributions spatiales
//

  imax=201;

  acc=1.;

// k_NL = 0 diffusivite constante -> EDP lineaire 
// k_NL = 1 diffusivite fonction de T -> EDP non-lineaire

  k_NL=1;

  itermax=16000;
  
  freq_dist=2000;
  
 // nombre de modes de Fourier utilises pour construire la solution exacte
 
  nmode=2;

// choix du pas d'avancement en temps adimensionne (pour solution exacte seulement)

  dFo  = 0.000051;

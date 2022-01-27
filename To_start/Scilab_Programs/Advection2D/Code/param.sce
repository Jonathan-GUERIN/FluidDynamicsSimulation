// parametres de la simulation
//
// physiques
//
//  a,b = vitesses d'onde
//  xmin,ymin, lx_dom, ly_dom : définissent le domaine de calcul
//  
// numeriques
//
// imax, jmax : Nombre de cellules de calcul
// cfl : nombre CFL
// itermax : Nombre d'iterations a effectuer
// freq_ech : Fréquence des sorties graphiques
//
// Vitesse d'onde
a = 0.;
b = 1;

// Abcisse minimale
xmin=0;
// Ordonnée minimale
ymin=0;

// Longueur du domaine suivant x
l_dom_x=2.0;

// Longueur du domaine suivant y
l_dom_y=1.0;

// Nombre de cellules
imax=41;
jmax=21;

// Nombre CFL
cfl=0.5;

// Choix de l'initialisation
// iinlet=0 (créneau)
iinlet=0;
// Choix du schéma
// 1 (schéma CIR ou de Roe)  2 (LW)
ischema=2;

// Frequence des sorties
freq_ech=1;

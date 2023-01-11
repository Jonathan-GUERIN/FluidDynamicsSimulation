
#include <iostream>
#include <cmath>
#include <fstream>
#include <string>

using namespace std;


double solex(double tau, double eta, int nmax)
{
	//calcul de la solution exacte

	double sumtheta = 0.;
	int nn = 0;
	for( int n = 0; n < nmax; n++)
	{
		nn = 2*n+1;
		sumtheta += (1./nn)*exp(-tau*pow((nn*M_PI/2.),2))*sin(nn*M_PI*eta/2.);
	}
	return 4/M_PI*sumtheta;
}

double cerror(double *th, double *thex, int imax)
{
	//calcul de l'erreur quadratique entre la simulation et la solution exacte 
	double e = 0.;

	for( int i = 0; i < imax; i++)
		e += pow((th[i]-thex[i]),2);

	return sqrt(e)/imax;
}

double min(double *t, int imax)
{
	//calul du minimum d'un tableau ( utilise par stabilise )
	double min = t[0];

	for( int i = 1; i < imax; i++)
	{
		if (t[i] < min)
			min = t[i];
		
	}

	return min;
}

double max(double *t, int imax)
{
	//calul du maximum d'un tableau ( utilise par stabilise )
	double max = t[0];

	for( int i = 1; i < imax; i++)
	{
		if (t[i] > max)
			max = t[i];	
	}

	return max;
}

int stabilite(double *th,int imax)
{
	// renvoie 0 si la simulation est instable et 1 si elle l'est
	if ((min(th,imax) < 0.) or ( max(th,imax) > 1.))
		return 0;

	return 1;
}

/*
void affiche(double *t, int imax)
{
	// permet d'afficher un tableau dans la console en cas de besoin
	for( int i = 0; i < imax; i++)
		cout << t[i] << " ";
	cout << endl;
}
*/

int main()
{
	//entree manuelle des parametres de la simulation
	
	double borne_alpha_inf=0.;
	double borne_alpha_sup=0.;
	int borne_imax_inf=0;
	int borne_imax_sup=0;

	cout << " alpha min : " << endl;
	cin >> borne_alpha_inf;
	cout << " alpha max : " << endl;
	cin >> borne_alpha_sup;
	cout << " imax min : " << endl;
	cin >> borne_imax_inf;
	cout << " imax max : " << endl;
	cin >> borne_imax_sup;

	cout << borne_alpha_inf <<endl;
	cout << borne_alpha_sup <<endl;

	//initialisation des variables

	int tmax = 500;

	double alpha = 0.00001;

	int imax = 100;

	double dt = 10.;

	int nb_iter = 0;

	string str = "";
	double df = 0.;
	double df1;
	double df2;

	int nmode = 10;

	double dx = 2./(imax-1);
	double xi[imax];
	double theta[imax];
	double flux_num[imax];
	double thetaex[imax];

	double Fo = 0.;
	double dFo = alpha*dt;

	
	double error = 1000.;
	int stable = 0;

	cout << std::scientific;

	//boucle de generation

	for( int a = 0; a < 10; a++)
	{
		alpha = borne_alpha_inf + (borne_alpha_sup - borne_alpha_inf)*a/10;
		cout << a << endl;
		for (int imax = borne_imax_inf ; imax < borne_imax_sup ; imax += 10)
		{
			cout << "	" << imax <<endl;
			df = 2./pow(imax+1,2)/alpha;
			df1 = 0.8*df;
			df2 = 1.2*df;
			for( int q = 0; q < 10; q++)
			{
				cout << "		" << dt << endl;
				dt = df1 + (df2-df1)*q/10;
				// dxi_theta
				Fo = 0.;
	
				for( int k = 0; k < imax ; k++)
					xi[k] = (k-1)*dx;

				
				for( int k = 0; k < imax; k++)
					theta[k] = 1;
				theta[0] = 0;
				theta[imax-1] = 0;

				// boucle principale

					//init
				nb_iter = (int)(tmax/dt);

				for( int j = 0; j < imax; j++)
					flux_num[j] = 0;

				error = 1000.;
				stable = 0;
					//simulation
				for( int j = 1; j < nb_iter+1; j++)
				{
					
					
					Fo += dFo;
					flux_num[0] = 0;

					for( int i = 1; i < imax; i++)
						flux_num[i] = (theta[i] - theta[i-1])/dx;


					for( int i = 1; i < imax-1; i++)
						theta[i] += (dFo/dx)*(flux_num[i+1]-flux_num[i]);

					for( int i = 0; i < imax; i++)
						thetaex[i] = solex(Fo,xi[i],nmode);
				}

				
					//calcul erreur et stabilite
				error = cerror(theta,thetaex,imax);
				
				stable = stabilite(theta, imax);
				
				// stockage dans un chaine de caractere
				str = str + to_string(tmax) + ";" + to_string(alpha) + ";" + to_string(dx) + ";" + to_string(dt) + ";" +to_string(stable) + ";" + to_string(nb_iter) + ";" + to_string(error) + "!";


			}
		}
	}

	//stockage des valeurs dans un fichier texte
	ofstream file("db.txt");
	
	file << str << endl;

	file.close();
	
	return 0;
}
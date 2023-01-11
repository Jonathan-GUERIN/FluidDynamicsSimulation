
import sqlite3
#ouverture du fichier texte
fichier = open("db.txt", "r")
#ouverture de la base de donnees
table = sqlite3.connect("db_c.db")
cursor = table.cursor()

#fonction d'ajout a la base de donnee
def ajoute_table(tm,al,dx,dtau,stable,nb_iter,error):
	cursor.execute("INSERT INTO db_diffusion1D_ VALUES ({},{},{},{},{},{},{})".format(tm,al,dx,dtau,stable,nb_iter,error))
	

#initialisation
L = []
T = []
temp = ""
cpt = 0
fichier2 = fichier.readlines()
fichier2 = fichier2[0]

#lecture du fichier
for k in fichier2:
	
	cpt = cpt%7

	
	if k == ";":
		#filtrage des infinis ( on choisit 10000000 de maniere arbitraire. cela n'a pas d'influence dans notre cas car on n'uyilise pas des simulations a valeur trop forte )
		if temp == "inf":
			temp ="10000000"

		if cpt in [0,4,5]:
			T.append(int(float(temp)))
		else:
			T.append(float(temp))
			
		temp = ""
		cpt += 1
	elif k == "!":
		#filtrage des infinis ( on choisit 10000000 de maniere arbitraire. cela n'a pas d'influence dans notre cas car on n'uyilise pas des simulations a valeur trop forte )
		if temp == "inf":
			temp ="10000000"
		if cpt in [0,4,5]:
			T.append(int(float(temp)))
		else:
			T.append(float(temp))
		L.append(list(T))
		T = []
		temp = ""
		cpt += 1
	else: 
		temp += k
	
#ajout a la base de donnees
for k in L:
	
	ajoute_table(k[0],k[1],k[2],k[3],k[4],k[5],k[6])

#fermeture des fichiers
table.commit()
table.close()
fichier.close()
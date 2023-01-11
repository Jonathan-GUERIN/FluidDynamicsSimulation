import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sqlite3




db = sqlite3.connect("Petite_table.db")

cur = db.cursor()


ligneall = cur.execute("SELECT * FROM A")

L = ligneall.fetchall()




F1 = []
F2 = []
F3 = []
F4 = []


for k in L :
	F1.append(k[9])
	F2.append(k[10])
	F3.append(k[11])
	F4.append(k[12])

plt.hist(F1,bins = 100)
plt.show()
plt.hist(F2,bins = 100)
plt.show()
plt.hist(F3,bins = 100)
plt.show()
plt.hist(F4,bins = 100)
plt.show()



db.close()
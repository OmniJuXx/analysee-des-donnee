import numpy as np

def mykmeans(X,k,maxit= 1000):
    
    # kmeans minimaliste qui deroule sur maxit iterations 
    # regle d arret  : stop apres  maxit iterations, pas tres malin
    
    # dimensions :
    n,p = np.shape(X)  
    
    # dimensionnement de la boite pour les donnees
    maxX = np.amax(abs(X),axis = 0) 	 
    # amax renvoie le max (ici de chaque colonne car axis =0)
    
    # initialisation : on genere k centres aleatoirements dans une 
    # boite parametree par maxX
    
    #  matrice k x p uniforme  dans [-1,1]
    Centres = 2 * np.random.rand(k,p) - 1
    # puis ans chaque direction on dilate ensuite de la quantite
    # maxX[j] : 
    Centres = np.dot(Centres,np.diag(maxX))  # np.dot : produit mat
    # vecteur pour stocker les num de classes    
    Groupes = np.zeros(n)
   
    # boucle de l algorithme kmeans
    for iter in range(maxit):
        # boucle sur les donnees : calcul des distances et affectation
        for i in range(n):
            xi = X[i,]
            # je place les distances entre xi et les centres dans distancesToxi
            distancesToxi = [np.linalg.norm(xi - Centres[j,]) for j in range(k)]
            # np.linalg.norm : fonction pour calcluer la norme d'un vecteur
            #affectation de xi a son centre le plus proche :
            Groupes[i] = np.argmin(distancesToxi)

        # actualisation des centres
        for l in range (k):
            # vecteur des indices de la classe l :
            indl = [i for i in range(n)  if Groupes[i] == l]
            Centres[l,] = X[indl,].mean(axis =0)
        
    return Centres,Groupes
    
    
    
import numpy as np

def ClassicalMDS(D,p):
    "Blablab"
 
    # Matrice de dissimilarite au carre : 
    Drond2 = D**2
    
    # Matrice de centrage :
    n = len(D)                                                                      
    H = np.eye(n) - np.ones((n, n))/n

    # Matrice de Gram centree                                                                                
    B = - 1/2 * np.dot(np.dot(H,Drond2),H)
 
    # Diagonalisation :                                                                            
    valPropre, vecPropre = np.linalg.eigh(B)
    # vecPropre : vect propres en colonnes

    # On selectionne les indices des p plus grandes valeurs propres 
    # elles sont ordonnees par ordre croissant                                              
    idDecr   = range(n-1,n-p-1,-1)
    valPropre = valPropre[idDecr]
    vecPropre = vecPropre[:,idDecr]

 
    # Configuration CMDS dans R^p :   
    diagSqrtVp  = np.diag(np.sqrt(valPropre))                  
    configCMDS = np.dot(vecPropre,diagSqrtVp)
 
    return configCMDS

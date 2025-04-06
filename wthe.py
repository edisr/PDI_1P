

import numpy as np

def calcula_wthe(image):

    v=0.5
    Pl=0.001
    r=1.0

    hist , _ = np.histogram(image.flatten(), 256, [0, 256])
    P = hist / hist.sum()  # PDF

    #maximo valor del histograma
    P_max = np.max(P)
    
    # Calcula Pu 
    Pu = v * P_max

    # Aplica la transformacion  Ω(P(k))
    P_wt = np.zeros_like(P)

    for k in range(256):
        if P[k] > Pu:
            P_wt[k] = Pu
        elif Pl <= P[k] <= Pu:
            P_wt[k] = (((P[k] - Pl) / (Pu - Pl)) ** r) * Pu
        else:
            P_wt[k] = 0

    # Compute the weighted histogram (transformed)
    H_wt = (256 - 1) * P_wt
    CDF = np.cumsum(H_wt)
    CDF = np.clip(CDF, 0, 255).astype('uint8')

    # Map the image using the CDF
    result = CDF[image]
    
    return result



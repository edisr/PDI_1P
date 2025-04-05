# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import cv2
import numpy as np
from skimage.measure import shannon_entropy
import pandas as pd

def calcula_ambe(image1, image2):
    """ Error de Brillo Medio Absoluto (AMBE) """
    return np.abs(np.mean(image1) - np.mean(image2))

def calcula_psnr(image1, image2):
    """ Peak Signal-to-Noise Ratio (PSNR) """
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calcula_entropia(image):
    """ Entropía """
    return shannon_entropy(image)

def calcula_contraste(image):
    """ Contraste (Desviación Estándar) """
    return np.std(image)

# Leer la imagen en escala de grises
image_gray = cv2.imread("C:\\Users\\Adis8\\Downloads\\test\\2.jpg", cv2.IMREAD_GRAYSCALE)

# Verificar si la imagen fue cargada
if image_gray is None:
    print("Error: No se pudo cargar la imagen")
else:
    # Equalizar el histograma de la imagen
    equalized_image = cv2.equalizeHist(image_gray)
    print("hoa")
    # Aplicar CLAHE imagen
    clahe = cv2.createCLAHE()
    cl1 = clahe.apply(image_gray)
    
    # Mostrar la imagen original,la imagen equalizada y por CLAHE
    cv2.imshow('Original Image', image_gray)
    cv2.imshow('Equalized Image', equalized_image)
    cv2.imshow('Clahe', cl1)
    
   

    datos = {
        "Parámetro": ["AMBE", "PSNR", "Entropía", "Contraste"],
        "Imagen ecualizada": [calcula_ambe(image_gray, equalized_image),
                              calcula_psnr(image_gray, equalized_image),
                              calcula_entropia(equalized_image),
                              calcula_contraste(equalized_image)],
        "Imagen ecualizada por clahe": [calcula_ambe(image_gray, cl1),
                                       calcula_psnr(image_gray, cl1),
                                       calcula_entropia(cl1),
                                       calcula_contraste(cl1)]
    }

    # Crear tabla con pandas
    tabla = pd.DataFrame(datos)
    
    # Imprimir la tabla
    print(tabla)
    tabla.to_csv("tabla_comparacion.csv", index=False)
    
    # Esperar una tecla y cerrar ventanas
    cv2.waitKey(0)
    cv2.destroyAllWindows()

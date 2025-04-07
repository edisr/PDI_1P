
import cv2
import numpy as np
from skimage.measure import shannon_entropy
import pandas as pd
from wthe import calcula_wthe
from dibujo_hist import dibujar_histograma
import matplotlib.pyplot as plt

def comparar_errores(datos):
    # Configuración para el gráfico
    parametros = datos["Parámetro"]
    valores_equalizada = datos["Imagen ecualizada"]
    valores_clahe = datos["Imagen ecualizada por CLAHE"]
    valores_wthe = datos["Imagen ecualizada por WTHE"]
    
    x = np.arange(len(parametros))  # Posición de los parámetros
    width = 0.25  # Ancho de las barras

    # Crear la figura
    plt.figure(figsize=(10, 6))
    
    # Graficar las barras
    plt.bar(x - width, valores_equalizada, width, label='Imagen ecualizada')
    plt.bar(x, valores_clahe, width, label='Imagen ecualizada por CLAHE', color='orange')
    plt.bar(x + width, valores_wthe, width, label='Imagen ecualizada por WTHE', color='green')
    
    # Etiquetas y título
    plt.xticks(x, parametros)
    plt.xlabel('Parámetro')
    plt.ylabel('Valores')
    plt.title('Comparación de parámetros por técnica de mejora de imagen')
    plt.legend()

    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()
    
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
    wthe_image= calcula_wthe(image_gray)
    # Aplicar CLAHE imagen
    clahe = cv2.createCLAHE()
    cl1 = clahe.apply(image_gray)
     # Aplicar WTHE a la imagen
    wthe_image= calcula_wthe(image_gray)
    
    # Mostrar la imagen original,la imagen equalizada y por CLAHE
     # Combinar las imágenes en una cuadrícula 2x2
    row1 = np.hstack((image_gray, equalized_image))  # Primera fila (arriba)
    row2 = np.hstack((cl1, wthe_image))             # Segunda fila (abajo)
    grid = np.vstack((row1, row2))                  # Unimos las dos filas
    
    # Muestra la cuadrícula completa
    cv2.imshow('Imagenes', grid)
    
    #Dibuja los histogramas para observar las diferencias
    dibujar_histograma(image_gray,equalized_image,cl1,wthe_image)
   

    datos = {
        "Parámetro": ["AMBE", "PSNR", "Entropía", "Contraste"],
        "Imagen ecualizada": [calcula_ambe(image_gray, equalized_image),
                              calcula_psnr(image_gray, equalized_image),
                              calcula_entropia(equalized_image),
                              calcula_contraste(equalized_image)],
        "Imagen ecualizada por clahe": [calcula_ambe(image_gray, cl1),
                                       calcula_psnr(image_gray, cl1),
                                       calcula_entropia(cl1),
                                       calcula_contraste(cl1)],
         "Imagen ecualizada por WTHE": [calcula_ambe(image_gray, wthe_image),
                                       calcula_psnr(image_gray, wthe_image),
                                       calcula_entropia(wthe_image),
                                       calcula_contraste(wthe_image)]
    }
    comparar_errores(datos)
    # Crear tabla con pandas
    tabla = pd.DataFrame(datos)
    
    # Imprimir la tabla
    print(tabla)
    tabla.to_csv("tabla_comparacion.csv", index=False)
    
    # Esperar una tecla y cerrar ventanas
    cv2.waitKey(0)
    cv2.destroyAllWindows()

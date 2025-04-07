
import cv2
from matplotlib import pyplot as plt

def dibujar_histograma(image_gray,equalized_image,cl1,wthe_image):

    # Calcula los histogramas para las imágenes
    hist_original = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
    hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
    hist_clahe = cv2.calcHist([cl1], [0], None, [256], [0, 256])
    hist_wthe = cv2.calcHist([wthe_image], [0], None, [256], [0, 256])
    
    # Configuración de la cuadrícula 2x2 para los gráficos
    plt.figure(figsize=(10, 10))
    
    # Histograma de la imagen original
    plt.subplot(2, 2, 1)
    plt.title("Original Image Histogram")
    plt.plot(hist_original, color='black')
    plt.xlim([0, 256])
    
    # Histograma de la imagen ecualizada
    plt.subplot(2, 2, 2)
    plt.title("Equalized Image Histogram")
    plt.plot(hist_equalized, color='blue')
    plt.xlim([0, 256])
    
    # Histograma de la imagen CLAHE
    plt.subplot(2, 2, 3)
    plt.title("CLAHE Histogram")
    plt.plot(hist_clahe, color='green')
    plt.xlim([0, 256])
    
    # Histograma de la imagen WTHE
    plt.subplot(2, 2, 4)
    plt.title("WTHE Histogram")
    plt.plot(hist_wthe, color='red')
    plt.xlim([0, 256])
    
    # Muestra la cuadrícula de histogramas
    plt.tight_layout()
    plt.show()
    

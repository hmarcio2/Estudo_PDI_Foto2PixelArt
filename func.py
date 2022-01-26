from numpy.lib.type_check import imag
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
import math
import random
import numpy as np
from copy import deepcopy
import cv2
import colorsys

def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h*w,3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
            clusters, 
            None,
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001), 
            rounds, 
            cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))


def quantizacao(img, K):
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    cv2.imshow('imagem quantizada', res2)
    cv2.waitKey(0)    
    return res2

def saturar_img(imagem):        
    nova_imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    for i in range(len(nova_imagem)):
        for j in range(len(nova_imagem[i])):            
            #saturação            
            if (nova_imagem[i][j][1] <= 32):
                nova_imagem[i][j][1] = 16
            elif (nova_imagem[i][j][1] > 32 and nova_imagem[i][j][1] <=64):
                nova_imagem[i][j][1] = 48
            elif (nova_imagem[i][j][1] > 64 and nova_imagem[i][j][1] <= 96):
                nova_imagem[i][j][1] = 80
            elif (nova_imagem[i][j][1] > 96 and nova_imagem[i][j][1] <= 128):
                nova_imagem[i][j][1] = 112
            elif (nova_imagem[i][j][1] > 128 and nova_imagem[i][j][1] <= 160):
                nova_imagem[i][j][1] = 144
            elif (nova_imagem[i][j][1] > 160 and nova_imagem[i][j][1] <= 192):
                nova_imagem[i][j][1] = 176
            elif (nova_imagem[i][j][1] > 192 and nova_imagem[i][j][1] <= 224):
                nova_imagem[i][j][1] = 208            
            else:
                nova_imagem[i][j][1] = 240
            #brilho
            if (nova_imagem[i][j][1] <= 32):
                nova_imagem[i][j][1] = 16
            elif (nova_imagem[i][j][1] > 32 and nova_imagem[i][j][1] <=64):
                nova_imagem[i][j][1] = 48
            elif (nova_imagem[i][j][1] > 64 and nova_imagem[i][j][1] <= 96):
                nova_imagem[i][j][1] = 80
            elif (nova_imagem[i][j][1] > 96 and nova_imagem[i][j][1] <= 128):
                nova_imagem[i][j][1] = 112
            elif (nova_imagem[i][j][1] > 128 and nova_imagem[i][j][1] <= 160):
                nova_imagem[i][j][1] = 144
            elif (nova_imagem[i][j][1] > 160 and nova_imagem[i][j][1] <= 192):
                nova_imagem[i][j][1] = 176
            elif (nova_imagem[i][j][1] > 192 and nova_imagem[i][j][1] <= 224):
                nova_imagem[i][j][1] = 208            
            else:
                nova_imagem[i][j][1] = 240
    nova_imagem = cv2.cvtColor(nova_imagem, cv2.COLOR_HSV2BGR)
    #cv2.imshow('imagem saturada raiz', nova_imagem)
    #cv2.waitKey(0)
    return nova_imagem

def saturar_img2(imagem, n_cores):      
    nova_imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)

    for i in range(len(nova_imagem)):
        for j in range(len(nova_imagem[i])):            
            #hue
            nova_imagem[i][j][0] = round(nova_imagem[i][j][0]/n_cores)*n_cores
            #saturação       
            if (nova_imagem[i][j][1] <= 32):
                nova_imagem[i][j][1] = 16
            elif (nova_imagem[i][j][1] > 32 and nova_imagem[i][j][1] <=64):
                nova_imagem[i][j][1] = 48
            elif (nova_imagem[i][j][1] > 64 and nova_imagem[i][j][1] <= 96):
                nova_imagem[i][j][1] = 80
            elif (nova_imagem[i][j][1] > 96 and nova_imagem[i][j][1] <= 128):
                nova_imagem[i][j][1] = 112
            elif (nova_imagem[i][j][1] > 128 and nova_imagem[i][j][1] <= 160):
                nova_imagem[i][j][1] = 144
            elif (nova_imagem[i][j][1] > 160 and nova_imagem[i][j][1] <= 192):
                nova_imagem[i][j][1] = 176
            elif (nova_imagem[i][j][1] > 192 and nova_imagem[i][j][1] <= 224):
                nova_imagem[i][j][1] = 208            
            else:
                nova_imagem[i][j][1] = 240
            #brilho
            if (nova_imagem[i][j][1] <= 32):
                nova_imagem[i][j][1] = 16
            elif (nova_imagem[i][j][1] > 32 and nova_imagem[i][j][1] <=64):
                nova_imagem[i][j][1] = 48
            elif (nova_imagem[i][j][1] > 64 and nova_imagem[i][j][1] <= 96):
                nova_imagem[i][j][1] = 80
            elif (nova_imagem[i][j][1] > 96 and nova_imagem[i][j][1] <= 128):
                nova_imagem[i][j][1] = 112
            elif (nova_imagem[i][j][1] > 128 and nova_imagem[i][j][1] <= 160):
                nova_imagem[i][j][1] = 144
            elif (nova_imagem[i][j][1] > 160 and nova_imagem[i][j][1] <= 192):
                nova_imagem[i][j][1] = 176
            elif (nova_imagem[i][j][1] > 192 and nova_imagem[i][j][1] <= 224):
                nova_imagem[i][j][1] = 208            
            else:
                nova_imagem[i][j][1] = 240
    nova_imagem = cv2.cvtColor(nova_imagem, cv2.COLOR_HSV2BGR)
    #cv2.imshow('imagem saturada raiz', nova_imagem)
    #cv2.waitKey(0)
    return nova_imagem

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = deepcopy(codebook[labels[label_idx]])
            label_idx += 1
    return image

def quantizar_cores_k_means(imagem, n):
    nova_imagem = np.array(imagem, dtype=np.float64)/255
    w,h,d = tuple(nova_imagem.shape)
    assert d == 3
    image_array = np.reshape(nova_imagem, (w*h,d))
    image_array_sample = shuffle(image_array, random_state = 0)[:1000]
    kmeans = KMeans(n_clusters = n, random_state=0).fit(image_array_sample)

    labels = kmeans.predict(image_array)
    codebook_random = shuffle(image_array, random_state=0)[:n]
    labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)
    nova_imagem = recreate_image(kmeans.cluster_centers_,labels,w,h)        
    cv2.imshow('imagem quantizada k-means', nova_imagem)
    cv2.waitKey(0)    
    nova_imagem = np.array(nova_imagem, dtype = np.float32)
    #input(nova_imagem[0:2])
    return nova_imagem

def filtro_mediana(imagem, k):
    nova_imagem = cv2.medianBlur(imagem, k)   

    cv2.imshow('imagem filtro mediana', nova_imagem)
    cv2.waitKey(0)
    return nova_imagem

def filtro_canny(imagem):
    canny = cv2.Canny(imagem, 250, 250) 
    nova_imagem = ~canny
    nova_imagem = cv2.merge((nova_imagem, nova_imagem, nova_imagem))       
    nova_imagem = cv2.addWeighted(imagem, 1.2, nova_imagem, -0.2, 25)    
    #cv2.imwrite('imagem do canny.jpg', ~canny)
    #cv2.imwrite('imagem com canny.jpg', nova_imagem)    
    return nova_imagem

def quantizar_cores(imagem, bits):
    nova_imagem = deepcopy(imagem)    
          
    nova_imagem = np.uint8(imagem/bits)*bits

    #cv2.imshow('imagem quantizada', nova_imagem)
    #cv2.waitKey(0)
    return nova_imagem

def quantizar_cores2(imagem, k):
    z =imagem.reshape((-1,3))
    z = np.float32(z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(z,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    nova_imagem = res.reshape((imagem.shape))
    
    #cv2.imwrite('imagem quantizada.jpg', nova_imagem)
    #cv2.imshow('imagem quantizada', nova_imagem)
    #cv2.waitKey(0)
    return nova_imagem

def mudar_paletaCores2(imagem):    
    nova_imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV)
    h =    [15,   5,   3,   0,  5,   10,  15,  25] 
    s =[0,  64,  89, 128, 153, 179, 204, 230, 255] # %[0  ,25 ,35,50,60,70,80,90,100]
    b =[255,255,230, 204, 179, 140, 102,  77,  64] # %[100,100,90,80,70,55,40,30, 25]

    for x in range(len(nova_imagem)):        
        for y in range(len(nova_imagem[x])):
            nova_imagem[x][y][0] = nova_imagem[x][y][0]/15
            nova_imagem[x][y][0] = int(nova_imagem[x][y][0]*15)
            print("original {} ".format(nova_imagem[x][y]))
            #verificar se é branco
            if (nova_imagem[x][y][1] < 64 and nova_imagem[x][y][2] > 230):
                nova_imagem[x][y][1] = s[0]
                nova_imagem[x][y][2] = b[0]
                print('branco')
                break
            #luminosidade somada
            if nova_imagem[x][y][0] < 30 or (nova_imagem[x][y][0] >=135 and nova_imagem[x][y][0] <= 180 ):
                #print("luz somatorio")
                if(nova_imagem[x][y][2] > 230):
                    nova_imagem[x][y][0] += h[0]                                       
                    nova_imagem[x][y][1] = s[1]
                    nova_imagem[x][y][2] = b[1]
                elif(nova_imagem[x][y][2] > 204 and nova_imagem[x][y][2] <= 230):
                    nova_imagem[x][y][0] += h[1]                    
                    nova_imagem[x][y][1] = s[2]
                    nova_imagem[x][y][2] = b[2]
                elif(nova_imagem[x][y][2] > 179 and nova_imagem[x][y][2] <= 204):
                    nova_imagem[x][y][0] += h[2]                    
                    nova_imagem[x][y][1] = s[3]
                    nova_imagem[x][y][2] = b[3]
                elif(nova_imagem[x][y][2] > 140 and nova_imagem[x][y][2] <= 179):
                    nova_imagem[x][y][0] += h[3]                    
                    nova_imagem[x][y][1] = s[4]
                    nova_imagem[x][y][2] = b[4]
            #luminosidade subtraida
            elif nova_imagem[x][y][0] > 30 and nova_imagem[x][y][0] < 135:
                print("luz subtraida")
                if(nova_imagem[x][y][2] > 230):                    
                    nova_imagem[x][y][0] -= h[0]                                        
                    nova_imagem[x][y][1] = s[1]
                    nova_imagem[x][y][2] = b[1]
                elif(nova_imagem[x][y][2] > 204 and nova_imagem[x][y][2] <= 230):
                    nova_imagem[x][y][0] -= h[1]
                    nova_imagem[x][y][1] = s[2]
                    nova_imagem[x][y][2] = b[2]
                elif(nova_imagem[x][y][2] > 179 and nova_imagem[x][y][2] <= 204):
                    nova_imagem[x][y][0] -= h[2]
                    nova_imagem[x][y][1] = s[3]
                    nova_imagem[x][y][2] = b[3]
                elif(nova_imagem[x][y][2] > 140 and nova_imagem[x][y][2] <= 179):
                    nova_imagem[x][y][0] -= h[3]
                    nova_imagem[x][y][1] = s[4]
                    nova_imagem[x][y][2] = b[4]                
            #cor central
            else:
                print("luz central")
                if(nova_imagem[x][y][2] > 230):                    
                    nova_imagem[x][y][1] = s[1]
                    nova_imagem[x][y][2] = b[1]
                elif(nova_imagem[x][y][2] > 204 and nova_imagem[x][y][2] <= 230):                    
                    nova_imagem[x][y][1] = s[2]
                    nova_imagem[x][y][2] = b[2]
                elif(nova_imagem[x][y][2] > 179 and nova_imagem[x][y][2] <= 204):                    
                    nova_imagem[x][y][1] = s[3]
                    nova_imagem[x][y][2] = b[3]
                elif(nova_imagem[x][y][2] > 140 and nova_imagem[x][y][2] <= 179):                    
                    nova_imagem[x][y][1] = s[4]
                    nova_imagem[x][y][2] = b[4]  
                            
            #sombra somatorio
            if nova_imagem[x][y][0] >= 45 and nova_imagem[x][y][0] < 120:
                print('sombra somada')
                if(nova_imagem[x][y][2] >= 140 and nova_imagem[x][y][2] < 179):
                    nova_imagem[x][y][0] += h[4]
                    nova_imagem[x][y][1] = s[5]
                    nova_imagem[x][y][2] = b[5]
                elif(nova_imagem[x][y][2] >= 102 and nova_imagem[x][y][2] < 140):
                    nova_imagem[x][y][0] += h[5]
                    nova_imagem[x][y][1] = s[6]
                    nova_imagem[x][y][2] = b[6]
                elif(nova_imagem[x][y][2] >= 77 and nova_imagem[x][y][2] < 102):
                    nova_imagem[x][y][0] += h[6]
                    nova_imagem[x][y][1] = s[7]
                    nova_imagem[x][y][2] = b[7]
                elif nova_imagem[x][y][2] < 77:
                    nova_imagem[x][y][0] += h[7]
                    nova_imagem[x][y][1] = s[8]
                    nova_imagem[x][y][2] = b[8]
            #soma subtração
            elif nova_imagem[x][y][0] < 45 or nova_imagem[x][y][0] > 120:
                print('sombra subtraida')
                if(nova_imagem[x][y][2] >= 140 and nova_imagem[x][y][2] < 179):
                    nova_imagem[x][y][0] -= h[4]
                    nova_imagem[x][y][1] = s[5]
                    nova_imagem[x][y][2] = b[5]
                elif(nova_imagem[x][y][2] >= 102 and nova_imagem[x][y][2] < 140):
                    nova_imagem[x][y][0] -= h[5]
                    nova_imagem[x][y][1] = s[6]
                    nova_imagem[x][y][2] = b[6]
                elif(nova_imagem[x][y][2] >= 77 and nova_imagem[x][y][2] < 102):
                    nova_imagem[x][y][0] -= h[6]
                    nova_imagem[x][y][1] = s[7]
                    nova_imagem[x][y][2] = b[7]
                elif nova_imagem[x][y][2] < 77:
                    nova_imagem[x][y][0] -= h[7]
                    nova_imagem[x][y][1] = s[8]
                    nova_imagem[x][y][2] = b[8]
            #cor central
            else:
                print('sombra central')
                if(nova_imagem[x][y][2] >= 140 and nova_imagem[x][y][2] < 179):                    
                    nova_imagem[x][y][1] = s[4]
                    nova_imagem[x][y][2] = b[4]
                elif(nova_imagem[x][y][2] >= 102 and nova_imagem[x][y][2] < 140):                    
                    nova_imagem[x][y][1] = s[5]
                    nova_imagem[x][y][2] = b[5]
                elif(nova_imagem[x][y][2] >= 77 and nova_imagem[x][y][2] < 102):                    
                    nova_imagem[x][y][1] = s[6]
                    nova_imagem[x][y][2] = b[6]
                elif nova_imagem[x][y][2] < 77:                    
                    nova_imagem[x][y][1] = s[7]
                    nova_imagem[x][y][2] = b[7]            
            
            input("alterado {} \n".format(nova_imagem[x][y]))

    nova_imagem = cv2.cvtColor(nova_imagem, cv2.COLOR_HSV2RGB)
    cv2.imshow('imagem com nova paleta de cor', nova_imagem)
    cv2.waitKey(0)
    
    return nova_imagem

def mudar_paletaCores(imagem):    
    nova_imagem = cv2.cvtColor(imagem, cv2.COLOR_RGB2HSV)
    pixel_color = []
    h = [0,8,15,23,30,38,45,53,60,68,75,83,90,98,105,113,120,128,135,143,150,158,165,173]
    intervalo_h = [4,11,19,26,34,41,49,56,64,71,79,86,94,101,109,116,124,131,139,146,154,161,169,176]
    sb = [0,84,192,255] #[0,84,169,255]
    b = [56,128,192,255]
    intervalo_sb = [26,55,127,212]
    
    for x in range(len(nova_imagem)):
        for y in range(len(nova_imagem[x])):
            #quantizar h
            pixel = [0,0,0]
            if(nova_imagem[x][y][0] > intervalo_h[-1] and (nova_imagem[x][y][0] <= intervalo_h[0])):
                pixel[0] = 0
            elif(nova_imagem[x][y][0] > intervalo_h[0] and (nova_imagem[x][y][0] <= intervalo_h[1])):
                pixel[0] = 1
            elif(nova_imagem[x][y][0] > intervalo_h[1] and (nova_imagem[x][y][0] <= intervalo_h[2])):
                pixel[0] = 2
            elif(nova_imagem[x][y][0] > intervalo_h[2] and (nova_imagem[x][y][0] <= intervalo_h[3])):
                pixel[0] = 3
            elif(nova_imagem[x][y][0] > intervalo_h[3] and (nova_imagem[x][y][0] <= intervalo_h[4])):
                pixel[0] = 4
            elif(nova_imagem[x][y][0] > intervalo_h[4] and (nova_imagem[x][y][0] <= intervalo_h[5])):
                pixel[0] = 5
            elif(nova_imagem[x][y][0] > intervalo_h[5] and (nova_imagem[x][y][0] <= intervalo_h[6])):
                pixel[0] = 6
            elif(nova_imagem[x][y][0] > intervalo_h[6] and (nova_imagem[x][y][0] <= intervalo_h[7])):
                pixel[0] = 7       
            elif(nova_imagem[x][y][0] > intervalo_h[7] and (nova_imagem[x][y][0] <= intervalo_h[8])):
                pixel[0] = 8    
            elif(nova_imagem[x][y][0] > intervalo_h[8] and (nova_imagem[x][y][0] <= intervalo_h[9])):
                pixel[0] = 9
            elif(nova_imagem[x][y][0] > intervalo_h[9] and (nova_imagem[x][y][0] <= intervalo_h[10])):
                pixel[0] = 10            
            elif(nova_imagem[x][y][0] > intervalo_h[10] and (nova_imagem[x][y][0] <= intervalo_h[11])):
                pixel[0] = 11   
            elif(nova_imagem[x][y][0] > intervalo_h[11] and (nova_imagem[x][y][0] <= intervalo_h[12])):
                pixel[0] = 12
            elif(nova_imagem[x][y][0] > intervalo_h[12] and (nova_imagem[x][y][0] <= intervalo_h[13])):
                pixel[0] = 13
            elif(nova_imagem[x][y][0] > intervalo_h[13] and (nova_imagem[x][y][0] <= intervalo_h[14])):
                pixel[0] = 14
            elif(nova_imagem[x][y][0] > intervalo_h[14] and (nova_imagem[x][y][0] <= intervalo_h[15])):
                pixel[0] = 15
            elif(nova_imagem[x][y][0] > intervalo_h[15] and (nova_imagem[x][y][0] <= intervalo_h[16])):
                pixel[0] = 16
            elif(nova_imagem[x][y][0] > intervalo_h[16] and (nova_imagem[x][y][0] <= intervalo_h[17])):
                pixel[0] = 17
            elif(nova_imagem[x][y][0] > intervalo_h[17] and (nova_imagem[x][y][0] <= intervalo_h[18])):
                pixel[0] = 18
            elif(nova_imagem[x][y][0] > intervalo_h[18] and (nova_imagem[x][y][0] <= intervalo_h[19])):
                pixel[0] = 19
            elif(nova_imagem[x][y][0] > intervalo_h[19] and (nova_imagem[x][y][0] <= intervalo_h[20])):
                pixel[0] = 20
            elif(nova_imagem[x][y][0] > intervalo_h[20] and (nova_imagem[x][y][0] <= intervalo_h[21])):
                pixel[0] = 21
            elif(nova_imagem[x][y][0] > intervalo_h[21] and (nova_imagem[x][y][0] <= intervalo_h[22])):
                pixel[0] = 22
            elif(nova_imagem[x][y][0] > intervalo_h[23]):
                pixel[0] = 23
            #quantizar S
            if(nova_imagem[x][y][1] <= intervalo_sb[0]):
                pixel[1] = 0          
            elif(nova_imagem[x][y][1] > intervalo_sb[1] and (nova_imagem[x][y][1] <= intervalo_sb[2])):
                pixel[1] = 1
            elif(nova_imagem[x][y][1] > intervalo_sb[2] and (nova_imagem[x][y][1] <= intervalo_sb[3])):
                pixel[1] = 2
            elif(nova_imagem[x][y][1] > intervalo_sb[3]):
                pixel[1] = 3
            #quantizar B
            if(nova_imagem[x][y][2] <= intervalo_sb[0]):
                pixel[2] = 0
            elif(nova_imagem[x][y][2] > intervalo_sb[1] and (nova_imagem[x][y][2] <= intervalo_sb[2])):
                pixel[2] = 1
            elif(nova_imagem[x][y][2] > intervalo_sb[2] and (nova_imagem[x][y][2] <= intervalo_sb[3])):
                pixel[2] = 2
            elif(nova_imagem[x][y][2] > intervalo_sb[3]):
                pixel[2] = 3
            

            #redução de cores
            if(pixel[2] == 1):
                pixel[1] = 3
            if(pixel[1] == 2 and pixel[2] == 2):
                pixel[2] = 3
            if(pixel[1] == 2 and pixel[2] == 3):
                pixel[1] = 3
                pixel[0] -= 1
            if(pixel[1] == 1 and pixel[2] == 3):
                pixel[1] = 2
                pixel[0] -= 1
            if(pixel[2] <= 0):
                pixel[1] = 0
            nova_imagem[x][y][0] = h[pixel[0]]
            nova_imagem[x][y][1] = sb[pixel[1]]
            #nova_imagem[x][y][2] = sb[pixel[2]]

    nova_imagem = cv2.cvtColor(nova_imagem, cv2.COLOR_HSV2RGB)

    cv2.imshow('imagem com nova paleta de cor', nova_imagem)
    cv2.waitKey(0)
    
    return nova_imagem

def equalizar_histograma(imagem):    
    r,g,b = np.float32(cv2.split(imagem)).astype(np.uint8)
    clahe_r = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_b = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    nova_imagem_r = clahe_r.apply(r)
    nova_imagem_g = clahe_g.apply(g)
    nova_imagem_b = clahe_b.apply(b)

    nova_imagem = cv2.merge((nova_imagem_r, nova_imagem_g, nova_imagem_b))
    
    #cv2.imwrite("imagem equalizada.jpg", nova_imagem)
    #cv2.imshow('imagem equalizada', nova_imagem)
    #cv2.waitKey(0)
    return nova_imagem

def limiarizacao(imagem):

    nova_imagem = cv2.medianBlur(imagem,7)
    #nova_imagem = cv2.adaptiveThreshold(imagem,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    ret,nova_imagem = cv2.threshold(imagem,128,255,cv2.THRESH_BINARY_INV)
    #nova_imagem = cv2.adaptiveThreshold(imagem,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    
    #cv2.imshow('imagem original', imagem)
    #cv2.imshow('imagem limiarizada', nova_imagem)
    cv2.waitKey(0)
    return nova_imagem

def diminuir_pixel(imagem, n):
    tamanho = imagem.shape[0:2]  
    x = int(tamanho[1] * n)
    y = int(tamanho[0] * n )
    
    nova_imagem = cv2.resize(imagem, (x , y), interpolation= cv2.INTER_NEAREST)    
    nova_imagem = cv2.resize(nova_imagem, (tamanho[1] , tamanho[0]), interpolation= cv2.INTER_NEAREST)    
    
    return nova_imagem

def filtro_sobel(imagem):       
    gray = cv2.cvtColor(imagem, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale = 1, delta = 0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale = 1, delta = 0, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)    
    nova_imagem = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5,0)
    nova_imagem = limiarizacao(nova_imagem)
    nova_imagem = cv2.merge((nova_imagem,nova_imagem,nova_imagem))    
    #cv2.imshow('filtro de sobel', nova_imagem)
    #cv2.waitKey(0)
    return nova_imagem

def pixelizacao(imagem, n):    
    r,g,b = cv2.split(imagem)    
    r = imagem[::n,::n]   
    r = np.repeat(r,n,axis=0)
    r = np.repeat(r,n,axis=1).astype(imagem.dtype)
    nova_imagem = r    
    #cv2.imshow('imagem pixelizada', nova_imagem)
    #cv2.waitKey(0)
    return nova_imagem

def somar_imagens(imagem1, imagem2):
    tamanho = imagem1.shape[0:2]      
    x = int(tamanho[1])
    y = int(tamanho[0])
    nova_imagem2 = cv2.resize(imagem2, (x , y), interpolation= cv2.INTER_NEAREST)   
    nova_imagem = cv2.addWeighted(imagem1, 0.2, nova_imagem2, 0.8,0)
    #cv2.imshow('pixel com contorno', nova_imagem)
    #cv2.waitKey(0)
    return nova_imagem

def adicionar_highlights(imagem):    
    dst_gray, dst_color = cv2.pencilSketch(imagem,sigma_s=60, sigma_r=0.1, shade_factor=0.07)    
    nova_imagem = cv2.addWeighted(imagem, 1.0, dst_color, -0.1,20)   
    cv2.imshow('highlights', dst_color)
    #cv2.imshow('imagem sem highlights', imagem)
    #cv2.imshow('imagem com highlights', nova_imagem)
    #cv2.waitKey(0)
    return nova_imagem

def aplicar_contorno_sobel(imagem):    
    contorno = filtro_sobel(imagem)
    nova_imagem = cv2.addWeighted(imagem, 1.1, contorno,-0.1,0)
    #cv2.imshow('imagem original', imagem)
    #cv2.imshow('imagem do contorno', contorno)
    #cv2.imshow('imagem com contorno', nova_imagem)
    #cv2.waitKey(0)
    return nova_imagem

def aplicar_contorno_stylization(imagem):
    dst_gray, dst_color = cv2.pencilSketch(imagem,sigma_s=60, sigma_r=0.1, shade_factor=0.07)    
    contorno = cv2.stylization(imagem, sigma_s = 60, sigma_r= 1.6)      
    
    nova_imagem = cv2.addWeighted(imagem, 1.2, contorno, -0.2,-10)
    #cv2.imshow('imagem com contorno.jpg', nova_imagem)        
    nova_imagem = cv2.addWeighted(imagem, 1.2, dst_color, -0.2,20)   
    #cv2.imshow('imagem com dst.jpg', nova_imagem)    
    
    #cv2.imshow('imagem do contorno.jpg', contorno)    
    #cv2.imshow('imagem da dst.jpg', dst_color)    
    #cv2.waitKey(0)
    return nova_imagem

def aumentar_nitidez2(imagem, imagem_original):
    gaussian_3 = cv2.GaussianBlur(imagem_original, (11, 11), 10)
    nova_imagem = cv2.addWeighted(imagem, 1.3, gaussian_3, -0.3, -10)
    cv2.imwrite('aumentar nitidez.jpg', gaussian_3)    
    #cv2.imshow('imagem sem nitidez', imagem)
    #cv2.imshow('imagem com nitidez', nova_imagem)
    #cv2.waitKey(0)
    return nova_imagem

def aumentar_nitidez(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    nova_imagem = sharpened
    cv2.imshow('imagem sem nitidez', image)
    cv2.imshow('imagem com nitidez', nova_imagem)
    cv2.waitKey(0)
    return nova_imagem

def pintura_oleo(imagem):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,6))
    morph = cv2.morphologyEx(imagem, cv2.MORPH_OPEN, kernel)
    nova_imagem = cv2.normalize(morph,None,20,255,cv2.NORM_MINMAX)

    
    #cv2.imshow('imagem com oleo', nova_imagem)
    #cv2.waitKey(0)
    return nova_imagem

def aumentar_contraste(imagem):
    clahe = cv2.createCLAHE(clipLimit = 3.,tileGridSize=(8,8))
    lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2,a,b))
    nova_imagem = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    #cv2.imshow('imagem aumento de contraste', nova_imagem)
    #cv2.waitKey(0)
    return nova_imagem

def aumentar_contraste2(imagem, alpha, beta):
    nova_imagem = cv2.addWeighted(imagem, alpha, np.zeros(imagem.shape, imagem.dtype),0,beta)
    #cv2.imshow('imagem aumento de contraste', nova_imagem)
    #cv2.waitKey(0)
    return nova_imagem

def transformar_imagem(imagem, nome):
    #img = equalizar_histograma(imagem)
    img= pintura_oleo(imagem)
    img = aplicar_contorno_stylization(img)
    img = quantizar_cores2(img, 32)    
    img = diminuir_pixel(img, 0.2)  
    cv2.imwrite(nome,img)

#mudar_paletaCores([0,0,0])


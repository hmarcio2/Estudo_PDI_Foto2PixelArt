from func import *

arquivo = 'astronauta'
path = 'img'
nome = '{}/{}.jpg'.format(path,arquivo)

img_original = cv2.imread(nome)
sizepixel = 0.2
cores = 32
tamanho = img_original.shape[0:2]  
if(tamanho[1]>tamanho[0]):
    img = cv2.resize(img_original, (800 , 600), interpolation= cv2.INTER_NEAREST)    
    img_original = cv2.resize(img_original, (800 , 600), interpolation= cv2.INTER_NEAREST)    
else:
    img = cv2.resize(img_original, (600 , 800), interpolation= cv2.INTER_NEAREST)    
    img_original = cv2.resize(img_original, (600 , 800), interpolation= cv2.INTER_NEAREST)    

img = equalizar_histograma(img)
img = pintura_oleo(img)
img = aplicar_contorno_stylization(img) #aquarela
img = quantizar_cores2(img, cores)
img = diminuir_pixel(img, sizepixel) #porcentagem

#img = equalizar_histograma(img)
cv2.imshow('imagem final', img)
cv2.waitKey(0)
#img = np.array(img, dtype = np.float32)*255
nome_final = '{}/{} pixelado.jpg'.format(path,arquivo)
cv2.imwrite(nome_final, img)
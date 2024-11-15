import numpy as np
import tensorflow as tf # type: ignore
import matplotlib as mpl
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import cv2
import base64
import pandas as pd

(x_train, y_train), (x_test, y_test) =  mnist.load_data()

def tulosta(arr):
    pixelVals = ['   ', ',,,', '<<<', '$$$', '@@@']
    zeroRange = 51
    rowPrint = []
    for row in range(28):
        for column in range(28):
            if arr[row][column] == 255:
                print('@@@', end='')
            else:
                print(pixelVals[(arr[row][column]//zeroRange)] , end='')
        print('')


#matriisi png:ksi:
#grayscaleArr = x_train[0].astype(np.uint8)
#cv2.imwrite('grayscale_image.png', grayscaleArr)

def annetaanArvot(input, piiloitettut, tulos):
    #8 saraketta ja 10 riviä kun input=10, piiloitettu=8
    W1 = np.random.randn(input, piiloitettut) #muotoa rivit, sarakkeet
    b1 = np.zeros((piiloitettut, 1))
    W2 = np.random.randn(piiloitettut, tulos)
    b2 = np.zeros((tulos, 1))
    print(np.shape(W1), np.shape(b1), np.shape(W2), np.shape(b2))
    return W1, b1, W2, b2

def frontProp(x, W1, b1, W2, b2):
    Z1 = np.dot(W1.T, x) + b1 #ns. weights ja biases, joiden muoto on 784x128 eli piilotetun ja input leijerin yhteydet, jossa neuroni n ja k:n yhteys on W[k][n]
    A1 = np.maximum(0, Z1) #ReLU aktivaatio
    Z2 = np.dot(W2.T, A1) + b2
    A2 = np.exp(Z2)/np.sum(np.exp(Z2))
    return Z1, A1, Z2, A2

def vectorify(matrix):
    matrix = np.array(matrix)
    matrix = matrix/255 # Normalisoimme arvot 0,1 välille
    vector = matrix.reshape(-1, 1) # Teemme (784, 1)-muotosen matriisin 28x28-matriisista, käytännössä vektori
    print(np.shape(vector))
    return(vector)

def output(A2):
    A2=A2.T # Teemme siitä pitkulan
    print(A2)
    print(float(A2[0][1]))
    print(np.shape(A2))
    suurin = [0,0] # Ekana luku, sitten arvo, esim. [4, 1,321415e-02]
    for val in range(10):
        if A2[0][val] > suurin[1]:
            print(f"tapahtu {val}")
            print(suurin[1])
            print(A2[0][val])
            print(suurin)
            suurin = [[val],[A2[0][val]]]
            print(suurin)
    return suurin


def crossError(y, Softmax):
    #Käytämme categorical cross-entropy virhelaskentaa. Vaikka tuloksia on vain 1, käytämme One-hot encodingia muutoksia varten.
    m=len(Softmax)
    Onehot = np.zeros((m, 1))
    Onehot[y] = 1
    log_loss = -np.log(Softmax)
    print(log_loss)
    loss = np.sum(log_loss)/m
    return loss

def MSE(y, Softmax):
    print(y)
    print(Softmax)
    print(np.shape(Softmax))
    oneHot = np.zeros(10)
    oneHot[y-1] = 1
    errorSum = 0
    for i in range(len(Softmax)):
        print((oneHot[i]-Softmax[i])**2)
        errorSum += (oneHot[i]-i*Softmax[i])**2
    return(errorSum/len(oneHot))

#def backProp(X, Y, ReLU, Softmax, Loss):
    



W1, b1, W2, b2 = annetaanArvot(784, 128, 10)

L1, ReLU, L2, Softmax = frontProp(vectorify(x_train[9321]), W1, b1, W2, b2)

print("Muodot:")
print(np.shape(L1), np.shape(ReLU), np.shape(L2), np.shape(Softmax))

print(tulosta(x_train[9321]))

print(np.shape(L1), np.shape(ReLU), np.shape(W2), np.shape(L2), np.shape(Softmax))

print(f"layer 2:\n {L2}, \n activation:\n {Softmax}") 

answer, confidence = output(Softmax)
print(f"Arvaus {answer}, Varmuus{confidence}")

loss = MSE(y_train[932], Softmax)
print(f"loss: {loss}")
import numpy as np
import tensorflow as tf # type: ignore
import matplotlib as mpl
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist
import cv2
import base64
import pandas as pd
import datetime
import csv


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
    #print(np.shape(W1), np.shape(b1), np.shape(W2), np.shape(b2))
    return W1, b1, W2, b2

def frontProp(x, W1, b1, W2, b2):
    Z1 = np.dot(W1.T, x) + b1 # ns. weights ja biases, joiden muoto on 784x128 eli piilotetun ja input leijerin yhteydet, jossa neuroni n ja k:n yhteys on W[k][n]
    A1 = np.maximum(0, Z1) # ReLU aktivaatio
    Z2 = np.dot(W2.T, A1) + b2 
    A2 = np.exp(Z2)/np.sum(np.exp(Z2)) # Softmax aktivaatio
    return Z1, A1, Z2, A2

def vectorify(matrix):
    matrix = np.array(matrix)
    matrix = matrix/255 # Normalisoimme arvot 0,1 välille
    vector = matrix.reshape(-1, 1) # Teemme (784, 1)-muotosen matriisin 28x28-matriisista, käytännössä vektori
    #print(np.shape(vector))
    return(vector)

def output(A2):
    A2=A2.T # Teemme siitä pitkulan
    suurin = [0,0] # Ekana luku, sitten arvo, esim. [4, 1.321415e-02]
    for val in range(10):
        if A2[0][val] > suurin[1]:
            #print(f"tapahtu {val}")
            #print(suurin[1])
            #print(A2[0][val])
            #print(suurin)
            suurin = [[val],[A2[0][val]]]
            #print(suurin)
    return suurin


def crossError(y, Softmax):
    #Ehkä käytämme categorical cross-entropy virhelaskentaa. Vaikka tuloksia on vain 1, käytämme One-hot encodingia muutoksia varten.
    m=len(Softmax)
    Onehot = np.zeros((m, 1))
    Onehot[y] = 1
    log_loss = -np.log(Softmax)
    #print(f"loss: {log_loss}")
    loss = np.sum(log_loss)/m
    return loss, Onehot

def MSE(y, Softmax):
    #print(y)
    #print(Softmax)
    #print(np.shape(Softmax))
    oneHot = np.zeros(10)
    oneHot[y-1] = 1
    errorSum = 0
    for i in range(len(Softmax)):
        #print((oneHot[i]-Softmax[i])**2)
        errorSum += (oneHot[i]-i*Softmax[i])**2
    return(errorSum/len(oneHot))


def backProp(X, Y, ReLU, Softmax, W2):
    m = X.shape[0] # Tässä saamme batchin koon
    dZ2 = Softmax - Y # Tässä toteamme että del L / del W2 = A2 - Y_one-hot
    dZ2 /= m # Saamme batchin arvot normalisoitua 
    dW2 = np.dot(ReLU, dZ2.T) # Tässä laskemme arvon del L / del W2
    db2 = np.sum(dZ2, keepdims=True, axis=0)
    dA1 = np.dot(dZ2.T, W2.T)
    dZ1 = dA1 * (dA1 > 0) # Tässä on relun derivaatta kertaa viimesten termien ketju eli dA1
    dW1 = np.dot(X, dZ1)
    db1 = np.sum(dZ1, keepdims=True, axis=0)
    return dW1, db1.T, dW2, db2
    #Loss = 1/n * sum_n((y_^)-y)**2 tai Loss = -y*log(softmax)
    #softMax = np.exp(Z)/np.sum(np.exp(Z))
    #ensin del L / del A2 jossa t=softMax(zW+b)
    # Muista se jacobian matriisi softmaxia varten, lähde on raportissa
    # Olkoon Z2 = zW+b
    # del t / del a = softMax'(zW+b)
    # Saamme nyt W_2 ja b_2
    # del Z2 / del W_2 = z^T
    # del Z2 / del b_2 = 1
    # del Z2 / del z = W^T
    # z = ReLU(zW+b)
    # a = zW+b
    # del z / del a = ReLU'(zW+b)
    # Saamme nyt vikat painot
    # del a / del W = z^T
    # del a / del b = 1
    # Gradientti W_2: del L / del W_2 = del L / del t * del t / del a * del a / del W
    # Gradientti b_2: del L / del W_2 = del L / del t * del t / del a * del a / del b_2
    # Gradientti W_1: del L / del W_1 = del L / del t_2 * del t_2 / del a_2 * del a_2 / del z_2 * del z_2 / del a_1 * del a_1 / del W_1
    # Gradientti b_1: del L / del W_1 = del L / del t_2 * del t_2 / del a_2 * del a_2 / del z_2 * del z_2 / del a_1 * del a_1 / del b_1  

def gradientDescent(W1, b1, W2, b2, learningRate, dW1, db1, dW2, db2):
    #print(np.shape(W1), np.shape(b1), np.shape(W2), np.shape(b2))
    W1 -= dW1 * learningRate
    b1 -= db1 * learningRate
    W2 -= dW2 * learningRate
    b2 -= db2 * learningRate


def training_loop(X, Y, learningRate, epochs):
    start_time = datetime.datetime.now()
    W1, b1, W2, b2 = annetaanArvot(784, 128, 10)
    losses = []
    #print(W1, b1, W2, b2)
    #with open("training_NN.csv", mode="w", newline="") as file:
        #writer = csv.writer(file)
        #writer.writerow(["Epoch", "Loss"])
    for epoch in range(epochs):
        X = vectorify(x_train[epoch])
        L1, ReLU, L2, Softmax = frontProp(X, W1, b1, W2, b2)
        #print(Softmax)
        #print(np.shape(Softmax))
        #print("Muodot:")
        #print(np.shape(L1), np.shape(ReLU), np.shape(L2), np.shape(Softmax))

        #print(tulosta(x_train[epoch]))

        #print(np.shape(L1), np.shape(ReLU), np.shape(W2), np.shape(L2), np.shape(Softmax))

        #print(f"layer 2:\n {L2}, \n activation:\n {Softmax}") 

        
        answer, confidence = output(Softmax)
        #print(f"Arvaus {answer}, Varmuus{confidence}")

        loss, oneHot = crossError(y_train[epoch], Softmax)
        #print(f"loss: {loss}")
        #print(oneHot)
        dW1, db1, dW2, db2 = backProp(X, oneHot, ReLU, Softmax, W2)
        gradientDescent(W1, b1, W2, b2, learningRate, dW1, db1, dW2, db2)
        if epoch % 100 == 0:
            #losses.append((epoch, loss)) # Laittaa sen listaan
            #writer.writerow([epoch, loss]) # Laittaa sen csv tiedostoon
            print(f"Kierros: {epoch}, Loss: {loss}")
    end_time = datetime.datetime.now()
    print("Valmis")
    print(f"Kesto {end_time - start_time}")
    return W1, b1, W2, b2

def testing(X, Y, W1, b1, W2, b2, amount):
    accuracy = 0
    for iter in range(amount):
        X_iter = vectorify(X[iter])
        x, y, z, A2 = frontProp(X_iter, W1, b1, W2, b2)
        if output(A2)[0] == Y[iter]:
            accuracy += 1
    return(accuracy/amount)

W1, b1, W2, b2 = training_loop(x_train, y_train, 0.01, 60000)

x_test = np.reshape(x_test, (10000, 784))

tarkkuus = testing(x_test, y_test, W1, b1, W2, b2, 10000)
print(f"Tarkkuus: {tarkkuus}")

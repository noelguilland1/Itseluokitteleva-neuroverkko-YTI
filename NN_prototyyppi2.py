import numpy as np
from keras.datasets import mnist
import datetime
import csv

# ((60000, 28, 28), (60000, 1)), ((10000, 28, 28), (10000, 1))
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(x_train.shape[0], -1)  # Shape: (60000, 784)
x_test = x_test.reshape(x_test.shape[0], -1)    # Shape: (10000, 784)

otos = 5000 # Koko datasetti on 60 000

def one_hot_encode(y, num_classes=10):
    return np.eye(num_classes)[y]

y_train = one_hot_encode(y_train) # (60000, 10, 1)
y_test = one_hot_encode(y_test) # (10000, 10, 1)

def initialize_parameters(layer_sizes):
    parameters = {}
    L = len(layer_sizes)

    for l in range(1, L):
        #parameters['W' + str(l)] = np.random.randn(layer_sizes[l], layer_sizes[l - 1]) * np.sqrt(2 / layer_sizes[l - 1])
        #parameters['b' + str(l)] = np.zeros((layer_sizes[l], 1))
        #np.savetxt("NNparametersW" + str(l) + ".txt", parameters['W' + str(l)])
        #np.savetxt("NNparametersb" + str(l) + ".txt", parameters['b' + str(l)])
        parameters['W' + str(l)] = np.loadtxt("NNparametersW" + str(l) + ".txt").reshape(layer_sizes[l], layer_sizes[l-1])
        parameters['b' + str(l)] = np.loadtxt("NNparametersb" + str(l) + ".txt").reshape(layer_sizes[l], 1)

    return parameters

def forward_propagation(X, parameters):
    caches = {}
    A = X
    L = len(parameters) // 2
    
    for l in range(1, L):
        A_prev = A
        Z = np.dot(parameters['W' + str(l)], A_prev) + parameters['b' + str(l)]
        A = np.maximum(0, Z)  # ReLU activation
        caches['A' + str(l)] = A
        caches['Z' + str(l)] = Z
        
    ZL = np.dot(parameters['W' + str(L)], A) + parameters['b' + str(L)]
    AL = np.exp(ZL) / np.sum(np.exp(ZL), axis=0, keepdims=True)  # Softmax activation
    caches['A' + str(L)] = AL
    caches['Z' + str(L)] = ZL
    
    return AL, caches

def compute_loss(AL, Y):
    m = Y.shape[1]
    loss = -np.sum(Y * np.log(AL + 1e-8)) / m
    return loss

def backward_propagation(X, Y, parameters, caches):
    grads = {}
    m = X.shape[1]
    L = len(parameters) // 2
    AL = caches['A' + str(L)]
    
    dZL = AL - Y
    grads['dW' + str(L)] = np.dot(dZL, caches['A' + str(L-1)].T) / m
    grads['db' + str(L)] = np.sum(dZL, axis=1, keepdims=True) / m
    
    for l in reversed(range(1, L)):
        dA_prev = np.dot(parameters['W' + str(l+1)].T, dZL)
        dZ = dA_prev * (caches['Z' + str(l)] > 0)  # ReLU derivative
        grads['dW' + str(l)] = np.dot(dZ, (X if l == 1 else caches['A' + str(l-1)]).T) / m
        grads['db' + str(l)] = np.sum(dZ, axis=1, keepdims=True) / m
        dZL = dZ
    
    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]
    return parameters

# Train the neural network
def train_neural_network(X, Y, otos, layer_sizes, learning_rate, epochs, batch_size, threshold):
    parameters = initialize_parameters(layer_sizes)
    start_time = datetime.datetime.now()
    luokiteltu = False # Ettei luokittele montaa kertaa, ottaa aikaa ja tekee mallista epätarkan
    accuracies = np.zeros(epochs)
    losses = []
    #with open("ItseluokittelevaNN" + str(otos) + ".csv", mode="w", newline="") as file:
    #    writer = csv.writer(file)
    #    writer.writerow(["Epoch", "Tarkkuus"])

    # Vain 15000 tiedossa
    X_initial = X[:otos]
    Y_initial = Y[:otos]
    X_unlabeled = X[otos:] # 45000, 784
    Y_unlabeled = np.zeros((X_unlabeled.shape[0], 10)) # 45000, 10
    itseluokitteluBatch = len(X_unlabeled)//epochs

    for epoch in range(epochs):
        for i in range(0, X_initial.shape[0], batch_size):
            X_batch = X_initial[i:i + batch_size].T  # Transpose to shape (features, batch_size)
            Y_batch = Y_initial[i:i + batch_size].T  # Transpose to shape (classes, batch_size)
            AL, caches = forward_propagation(X_batch, parameters)
            loss = compute_loss(AL, Y_batch)
            grads = backward_propagation(X_batch, Y_batch, parameters, caches)
            parameters = update_parameters(parameters, grads, learning_rate)
        learning_rate *= 0.99
        accuracies[epoch] = test_neural_network(x_test.T, y_test.T, parameters) # Pidetään listaa kaikista tarkuuksista
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Accuracy: {accuracies[epoch]:.4f}")
        
        #with open("ItseluokittelevaNN" + str(otos) + ".csv", mode="a", newline="") as file:
        #    writer = csv.writer(file)
        #    losses.append((epoch, accuracies[epoch])) # Laittaa sen listaan
        #    writer.writerow([epoch, accuracies[epoch]]) # Laittaa sen csv tiedostoon


        # Itseloukittelu vaiheet:
        
        if epoch >= 2 and (accuracies[epoch-1] + accuracies[epoch-2])/2 > accuracies[epoch] and luokiteltu == False:
            luokiteltu = True
            print(f"Threshold accuracy achieved! Accuracy: {accuracies[epoch]}")
            X_initial, Y_initial = itseluokittelu(X_unlabeled, parameters, X_initial, Y_initial)

        X_initial, Y_initial = itseluokittelu(X_unlabeled[:int(itseluokitteluBatch*accuracies[epoch])], parameters, X_initial, Y_initial)
        X_unlabeled = X_unlabeled[int(itseluokitteluBatch*accuracies[epoch]):]

        print(np.shape(X_unlabeled))
    end_time = datetime.datetime.now()
    print("Training complete")
    print(f"Duration: {end_time - start_time}")
    return parameters


def itseluokittelu(X_unlabeled, parameters, X_initial, Y_initial):
    Y_arvot, _ = forward_propagation(X_unlabeled.T, parameters)
    predictions = np.argmax(Y_arvot, axis=0)
    Y_unlabeled = one_hot_encode(predictions)
    X_initial = np.concatenate((X_initial, X_unlabeled), axis=0)
    Y_initial = np.concatenate((Y_initial, Y_unlabeled), axis=0)
    return X_initial, Y_initial

def test_neural_network(X, Y, parameters):
    AL, _ = forward_propagation(X, parameters)
    predictions = np.argmax(AL, axis=0)
    labels = np.argmax(Y, axis=0)
    accuracy = np.mean(predictions == labels)
    return accuracy

# (input, hidden_1, ..., hidden_n, output)
layer_sizes = [784, 512, 256, 10]


parameters = train_neural_network(x_train, y_train, otos, layer_sizes, learning_rate=0.01, epochs=200, batch_size=128, threshold=0.9)


accuracy = test_neural_network(x_test.T, y_test.T, parameters)
print(f"Final Accuracy: {accuracy:.4f}")

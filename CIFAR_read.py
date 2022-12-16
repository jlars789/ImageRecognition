import pickle
import numpy as np

def readCIFAR(filename):
    filename = "cifar-10-batches-py/" + filename
    with open(filename, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    y = np.array(data[b'labels'])

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw_images, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, 3, 32, 32])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    X = images.reshape((images.shape[0], 3*32*32))
    return X, y

def CIFARimg(filename):
    filename = "cifar-10-batches-py/" + filename
    with open(filename, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')

    # Get the raw images.
    raw_images = data[b'data']
    raw_images = raw_images.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    labels = np.array(data[b'labels'])
    return raw_images, labels

def testCIFAR(X, Y, beta):
    z = np.dot(X, beta)
    #Logistic Regression function for predicted y values
    predY = (1/(1+np.exp(-z)))
    predY = np.around(predY)
   
    correct = [[], []]
    incorrect = [[], []]
    for i in range(len(X)):
        if predY[i] == Y[i]:
            correct[int(Y[i])].append(i)
        else:
            incorrect[int(Y[i])].append(i)
    return correct, incorrect
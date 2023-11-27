import argparse
import csv
import numpy as np


def calculateGradient(W, X, Y, Op, learningRate):
    gradient = (Y - Op) * X
    gradient = np.sum(gradient, axis=0)
    temp = np.array(learningRate * gradient).reshape(W.shape)
    W = W + temp
    return gradient, W


def calculateSSE(Y, Op):
    sse = np.sum(np.square(Op - Y))
    return sse


def calculatePredicatedValue(X, W):
    Op = np.dot(X, W)
    return Op

def main():
    args = parser.parse_args()
    file, learningRate, threshold = args.data, float(args.eta), float(args.threshold) 
    
    
    with open(file) as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        X = []
        Y = []
        for row in reader:
            X.append([1.0] + row[:-1])
            Y.append([row[-1]])
    
   
    n = len(X)
    X = np.array(X).astype(float)
    Y = np.array(Y).astype(float)
    W = np.zeros(X.shape[1]).astype(float)
    
    W = W.reshape(X.shape[1], 1).round(4)
    
    
    Op = calculatePredicatedValue(X, W)
    
    
    sse_old = calculateSSE(Y, Op)
    print(f"0,{','.join([f'{w:.9f}' for w in W.T[0]])},{sse_old:.9f}")
    gradient, W = calculateGradient(W, X, Y, Op, learningRate)
    
    i = 1
    while True:
        Op = calculatePredicatedValue(X, W)
        sse_new = calculateSSE(Y, Op)

        if abs(sse_new - sse_old) > threshold:
            print(f"{i},{','.join([f'{w:.9f}' for w in W.T[0]])},{sse_new:.9f}")
            gradient, W = calculateGradient(W, X, Y, Op, learningRate)
            i += 1
            sse_old = sse_new
        else:
            break
    print(f"{i},{','.join([f'{w:.9f}' for w in W.T[0]])},{sse_new:.9f}")





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="Data File")
    parser.add_argument("-l", "--eta", help="Learning Rate")
    parser.add_argument("-t", "--threshold", help="Threshold")
    main()
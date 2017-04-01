"""
	Project no.2
	student ID : 20113337
	student name : Choi_young keun

	Single-layer perceptron
"""

import numpy as np
import matplotlib.pyplot as plt
import random

"""-------------------------------------------------
	Name : plotTheDecisionBoundary()
	Input = 1*3 weight vector, state
	Description : ploting the linear equation of weight vector
-------------------------------------------------"""
def plotTheDecisionBoundary(weight, state):
	rangeX, rangeY = np.arange(-7,10), np.arange(-7,10)
	xx, yy = np.meshgrid(rangeX, rangeY)
	Z = weight[0]*xx + weight[1]*yy + weight[2]

	if state == 1:
		plt.contour(xx, yy, Z, [0.1], colors = 'm')
	elif state == 0:
		plt.contour(xx, yy, Z, [0.1], colors = 'y')

"""-------------------------------------------------
	Name : plotTheTwoData()
	Input = n*3 matrix
	Description : ploting the input data
-------------------------------------------------"""
def plotTheTwoData(w1, w2):
	plt.axis([-7, 10, -8, 10])
	for i in range(0, 10):
		plt.plot(w1[i][0], w1[i][1], 'rs')
		plt.plot(w2[i][0], w2[i][1], 'bo')

"""-------------------------------------------------
	Name : readFile()
	Input = n*3 matrix
	Description : read the file and save the data
-------------------------------------------------"""
def readFile(w1, w2):

	f = open("input.txt", 'r')
	lines = f.readlines()
	inputData = []

	for line in lines:
		word = line.split(' ')
		inputData.append(word)

	for i in range(0, 10):
		w1[i][0], w1[i][1] = inputData[i][0], inputData[i][1]
		w2[i][0], w2[i][1] = inputData[i][2], inputData[i][3]

	f.close()

"""-------------------------------------------------
	Name : perceptron()
	Input = n*3 matrix, 1*3 weight vector
	Description : input*weight -> layer -> output
				  output -> activate -> return f(net)
-------------------------------------------------"""
def perceptron(data, weight):
	
	temp = data[0]*weight[0] + data[1]*weight[1] + data[2]*weight[2]
	net = activateFunc(temp)

	return net

"""-------------------------------------------------
	Name : calculateErrorRate()
	Input = true value, predict value
	Description : calculate the f(net) value
-------------------------------------------------"""
def calculateErrorRate(value, predictValue):
	return abs((value - predictValue)/2)

"""-------------------------------------------------
	Name : activateFunc()
	Input = value
	Description : input data > 0  -> class 1
				  input data < 0  -> class 2
-------------------------------------------------"""
def activateFunc(sum):

	if sum > 0:
		return 1
	else:
		return -1


if __name__ == '__main__':
	
	w1Data, w2Data = np.ones([10, 3]), np.zeros([10, 3])

	readFile(w1Data, w2Data)
	weight = np.zeros(3)
	error = 0
	
	#learingRate is random(0~1)
	#weight vector random(-0.5~0.5)
	while(1):
		totalError = 0
		learningRate = random.random()
		weight[0] = random.randrange(-5, 6)/10.0
		weight[1] = random.randrange(-5, 6)/10.0
		weight[2] = random.randrange(-5, 6)/10.0

		for i in range(0, 2*len(w1Data)):
			if i < 10:
				error = calculateErrorRate(1, perceptron(w1Data[i], weight))
				weight[0] += learningRate*error*w1Data[i][0]
				weight[1] += learningRate*error*w1Data[i][1]
				weight[2] += learningRate*error
			else:
				error = calculateErrorRate(-1, perceptron(w2Data[i-10], weight))
				weight[0] += learningRate*error*w2Data[i-10][0]
				weight[1] += learningRate*error*w2Data[i-10][1]
				weight[2] += learningRate*error
			totalError += error
		
		print "--------------------------------------------------------"
		print "| learning Rate : ", learningRate
		print "| Error Rate : ", totalError
		print "| weight : ", weight[0], weight[1], weight[2]
		print "--------------------------------------------------------" 

		if totalError == 0:
			break
		elif totalError <= 2:
			plotTheDecisionBoundary(weight, 0)
	
	plotTheDecisionBoundary(weight, 1)	
	plotTheTwoData(w1Data, w2Data)
	plt.show()


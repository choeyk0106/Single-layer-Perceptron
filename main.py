"""
	Project no.2
	student ID : 20113337
	student name : Choi_young keun
"""

import numpy as np
import matplotlib.pyplot as plt
import random

def plotTheDecisionBoundary(weight, state):
	rangeX, rangeY = np.arange(-7,10), np.arange(-7,10)
	xx, yy = np.meshgrid(rangeX, rangeY)
	Z = weight[0]*xx + weight[1]*yy + weight[2]

	if state == 1:
		plt.contour(xx, yy, Z, [0.1], colors = 'm')
	elif state == 0:
		plt.contour(xx, yy, Z, [0.1], colors = 'y')

def plotTheTwoData(w1, w2):
	plt.axis([-7, 10, -8, 10])
	for i in range(0, 10):
		plt.plot(w1[i][0], w1[i][1], 'rs')
		plt.plot(w2[i][0], w2[i][1], 'bo')

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

def perceptron(data, weight):
	
	temp = data[0]*weight[0] + data[1]*weight[1] + data[2]*weight[2]
	net = activateFunc(temp)

	return net

def calculateErrorRate(value, predictValue):
	return abs((value - predictValue)/2)

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
	
	it = 0
	#while(it < 100):
	#	it += 1
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


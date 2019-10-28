from __future__ import print_function # to avoid issues between Python 2 and 3 printing

import numpy as np

from skimage import data, io, color, transform, exposure
from pprint import pprint
from mpl_toolkits.mplot3d import Axes3D
import math

import os
import sys
import csv
import re
import pandas as pd
from matplotlib import pyplot as plt
from utilities import *


x, y = load_points_from_file(sys.argv[1])

num_line_segments = len(x)/ 20

i=0
while (i<num_line_segments):
	locals()['x' + str(i+1) ]= x[(20*i):(20 *(i+1))]
	locals()['y' + str(i+1) ]= y[(20*i):(20 *(i+1))]
	locals()['x' + str(i+1) + '_min'] = locals()['x' + str(i+1) ].min()
	locals()['x' + str(i+1) + '_max' ] = locals()['x' + str(i+1) ].max()
	locals()['xx' + str(i+1) ] = np.linspace(locals()['x' + str(i+1) + '_min' ],locals()['x' + str(i+1) + '_max'],endpoint=True)
	i = i + 1


#linear
def least_squares(xi, yi):
	o = np.ones(xi.shape)
	X = np.column_stack((o, xi))
	A= np.linalg.inv((X.T.dot(X))).dot(X.T).dot(yi)
	return A

#polynomial3 for square
def least_squares2(x, y):
	o = np.ones(x.shape)
	X = np.column_stack((o, x,  x ** 2))
	B = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(y)
	return B
	#plot

#cubic
def least_squares3(x, y):
	o = np.ones(x.shape)
	X = np.column_stack((o, x,  x ** 2, x ** 3))
	A = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(y)
	return A
	#plot



#sin wave
# def least_squares4(x, y):
# 	o = np.ones(x.shape)
# 	X = np.column_stack((o, x.sin()))
# 	A = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(y)
# 	return A
# 	#plot

total_error = 0
i = 0
fig,ax= plt.subplots()
while(i < num_line_segments):
	X_1=locals()['x' + str(i+1) ]
	Y_1=locals()['y' + str(i+1) ]
	XX=locals()['xx' + str(i+1) ]
	a=least_squares(locals()['x' + str(i+1)],locals()['y' + str(i+1)])
	b=least_squares2(locals()['x' + str(i+1)],locals()['y' + str(i+1)])
	c=least_squares3(locals()['x' + str(i+1)],locals()['y' + str(i+1)])
	# d=least_squares4(locals()['x' + str(i+1)],locals()['y' + str(i+1)])

	j=0
	error1=0
	error2=0
	error3=0
	while j<20:
		mu_linear = a[0] + a[1] * X_1[j]
		error1+=pow((Y_1[j] - mu_linear),2)

   #polynomial error
		mu_polynomial = b[0] + b[1] * X_1[j] + b[2] * pow(X_1[j],2)
		error2 += pow((mu_polynomial - Y_1[j]), 2)

	#cubic error
		mu_polynomial2 = c[0] + c[1] * X_1[j] + c[2] * pow(X_1[j], 2) + c[3] * pow(X_1[j], 3)
		error3+= pow((mu_polynomial2 - Y_1[j]), 2)
		j+=1

	print('the first error is :', error1)
	print('the second error is :', error2)
	print('the third error is :', error3)
	# mu_polynomial3 = d[0] + d[1] * np.mean(X_1.sin())
	# error_sin = pow((mu_polynomial3- np.mean(Y_1)), 2)
	# print('the fourth error is :', error_sin)

	real_error=(error1, error2, error3)
	index1=real_error.index(min(real_error))
	total_error+=min(real_error)
	#linear
	if index1 == 0:
		real_error = a
		line = a[0] + a[1] * XX
		ax.plot(XX,line,c="g")


	#square
	if index1 == 1:
		real_error = b
		line2 = b[0] + b[1] * XX + b[2] * (XX ** 2)
		ax.plot(XX,line2,c="b")

	#cubic
	if index1 == 2:
		real_error = c
		line3 =  c[0] + c[1] * XX + c[2] * (XX ** 2) + c[3] * (XX ** 3)
		ax.plot(XX,line3,c="g")


	i+=1
	# if index1 == 3:
	# 	real_error = d
	# 	print('the real error is:', real_error)
	# 	locals()['error_total' + str(i+1)] = real_error.min()
	# 	plt.subplots()
	# 	line4 =  d[0] + d[1] * X_1.sin()
	# 	plt.scatter(X_1, Y_1)
	# 	plt.plot(X_1,line4)
	# 	plt.show()

ax.scatter(x,y)
plt.show()
print(total_error)


# def comparator(a,b):
# 	if a<b:
# 		return a
# 	return b
#
#
# # #unknown function sin
#
# def least_squares5(x, y):
# 	o = np.ones(x.shape)
# 	X = np.column_stack((o, sin(x)))
# 	A = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(y)
# 	return A
# 	#plot
#
# plt.subplots()
# xx = np.linspace(x_min,x_max,endpoint=True)
#
#
# plt.scatter()
# plt.plot(xx,)
# plt.show()

#factorial function
# def factorial(n):
# 	if n <= 0:
# 		return 1
# 	else:
# 		return n * factorial(n-1)
# #taylor approximation for the unknown print_function
# def taylor(function, x0, n):
# 	j=0
# 	p=0
# 	while j <= n:
# 		p = p + function(x, j).subs(x, x0)/factorial()*(x - x0) ** j
# 		i += 1
# 	return p














#error



#def view_data_segments(xs, ys):
#    """Visualises the input file with each segment plotted in a different colour.
#    Args:
#        xs : List/array-like of x co-ordinates.
#        ys : List/array-like of y co-ordinates.
#    Returns:
#        None
#    """
#    assert len(xs) == len(ys)
#    assert len(xs) % 20 == 0
#    len_data = len(xs)
#    num_segments = len_data // 20
#    colour = np.concatenate([[i] * 20 for i in range(num_segments)])
#    plt.set_cmap('Dark2')
#    plt.scatter(xs, ys, c=colour)
#    plt.show()

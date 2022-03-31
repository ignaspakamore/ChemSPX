import numpy as np
from numpy import pi 
from numpy import arange
from numpy import exp
from numpy import sqrt
from numpy import cos
from numpy import e
from numpy import pi

class Fx():
	def __init__(self, indict, train_data):
		self.indict = indict
		self.train_data = train_data
		
	def f_x(self, X):
		x = X[0]
		y = X[1]

		#ACKLEY FUNCTION
		fx = -20.0 * exp(-0.2 * sqrt(0.5 * (x**2 + y**2)))-exp(0.5 * (cos(2 * pi * x)+cos(2 * pi * y))) + e + 20

		return fx

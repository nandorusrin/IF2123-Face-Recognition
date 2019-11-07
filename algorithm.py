# Author: Nando Rusrin Pratama (13517148@std.stei.itb.ac.id)
# 
# Algorithm for Euclidean Distance and Cosine Similarity

import math

class Matcher(object):
	def euclidean_distance(x, y):
		distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]))
		return distance

	def cosine_similarity(x, y):
		similarity = sum([(a * b) for a, b in zip(x, y)]) / (math.sqrt(sum([a ** 2 for a in x])) * math.sqrt(sum([b ** 2 for b in y])))
		return similarity

def run():
	x = (5, 6, 7)
	y = (8, 9, 9)
	print euclidean_distance(x, y)
	print cosine_similarity(x, y)

run()
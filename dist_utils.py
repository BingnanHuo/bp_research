import numpy as np
import math


def dist(arr, pointA, pointB):
	return dist_point(arr[pointA], arr[pointB])

def dist_point(pointA, pointB):
	return np.sqrt(((pointA - pointB)**2).sum())


def height(arr, pointA, pointB):
	return np.abs(arr[pointA][1]-arr[pointB][1])

def width(arr, pointA, pointB):
	return np.abs(arr[pointA][0]-arr[pointB][0])

def angle(arr, pointA, pointB):
    # pointA should be the one on the left in image
	dy = arr[pointB][1]-arr[pointA][1]
	dx = arr[pointB][0]-arr[pointA][0]
	return math.degrees(math.atan2(-dy, dx)) # use -dy because in images, the y axis is downward

def slope(arr, pointA, pointB):
	dy = arr[pointB][1]-arr[pointA][1]
	dx = arr[pointB][0]-arr[pointA][0]
	return np.abs(dy/dx)

def find_max(A,B):
	return np.nanmax(np.array([A/B, B/A]))

def find_max_alt(A,B):
    # for cases with 0 division
    return np.nanmax(np.array([np.log((A+1)/(B+1)), np.log((B+1)/(A+1))]))

#def next_dist(pointA):
#	return dist()

def close_seg(arr, points):
	sum_dist = 0
	for i in points:
		sum_dist += dist(arr, i, i+1)
	sum_dist -= dist(arr, points[-1], points[-1]+1)
	sum_dist += dist(arr, points[0], points[-1])
	return sum_dist



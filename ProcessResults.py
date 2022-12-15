from turtle import distance
import numpy as np
import csv
from dist_utils import *
from landmark_utils import to_gemma_landmarks, rotate_landmarks


class ProcessResults:
     
    def __init__(self, input_landmark, mode=None):
        super().__init__()
        if mode is None:
            self._model = input_landmark._ModelName
            self._landmarks = to_gemma_landmarks(input_landmark._shape)
            self._lefteye = input_landmark._lefteye
            self._righteye = input_landmark._righteye
        else:
            self._landmarks = to_gemma_landmarks(input_landmark)
            self._lefteye = None
            self._righteye = None

        self._dists = {}
        self._features = {}
        self._features_arr = np.zeros((29),dtype=np.float32)


        self._physical_iris_diameter = 11.77 # Default Value
        self._pixel_scale = -1 # how many mm is one pixel measuring. multiply this scale to measurements in pixels to get physical distances.
        self._scaled_by_pixel = False
        
        self._bbox_dist = -1
        self._scaled_by_proportion = False
        
        self.correct_tilts()
        self.compute_distances()
        self.compute_features()
        self.features_to_arr()

    def correct_tilts(self):
        tilt_angle = angle(self._landmarks, 48, 49)
        self._landmarks = rotate_landmarks(self._landmarks,
                                           tuple(self._landmarks[48]),
                                           -tilt_angle)
        
    def compute_distances(self):
        self._dists = {
            'A':  width(self._landmarks, 48, 49),
            'Bl': width(self._landmarks, 10, 13),
            'Br': width(self._landmarks, 16, 19),
            'C' : dist(self._landmarks, 37, 50), # OR self._gemma_landmarks[50][1]-self._gemma_landmarks[37][1]
            'D' : width(self._landmarks, 10, 48),
            'E' : width(self._landmarks, 19, 49),
            'F' : dist(self._landmarks, 10, 37),
            'G' : dist(self._landmarks, 19, 37),
            'H' : dist(self._landmarks, 10, 23),
            'I' : dist(self._landmarks, 19, 27),
            'J' : dist(self._landmarks, 23, 37),
            'K' : dist(self._landmarks, 27, 37),
            'L' : np.average(self._landmarks[0:5][...,1]),
            'M' : np.average(self._landmarks[5:10][...,1]),
            'Nl': height(self._landmarks, 11, 15),
            'Nr': height(self._landmarks, 12, 14),
            'Ol': height(self._landmarks, 17, 21),
            'Or': height(self._landmarks, 18, 20),
            
            'Pl': height(self._landmarks, 29, 39),
            'Pu': height(self._landmarks, 30, 38),
            'Ql': height(self._landmarks, 32, 36),
            'Qu': height(self._landmarks, 33, 35),
            'R' : dist(self._landmarks, 3, 37),
            'S' : dist(self._landmarks, 6, 37),
            'T' : dist(self._landmarks, 2, 37),
            'U' : dist(self._landmarks, 7, 37),
            'Vl': dist(self._landmarks, 28, 37),
            'Vr': dist(self._landmarks, 34, 37),
            'W' : width(self._landmarks, 28, 34),
            'X' : dist(self._landmarks, 25, 31),
            'Wl': close_seg(self._landmarks, [28,29,30,31,37,38,39]),
            'Wr': close_seg(self._landmarks, [31,32,33,34,35,36,37])
        }
        self._dists['N'] = np.average((self._dists['Nl'], self._dists['Nr']))
        self._dists['O'] = np.average((self._dists['Ol'], self._dists['Or']))

        
    def compute_features(self):
        self._features = {
            # Eyebrows
            'f0' : np.abs(angle(self._landmarks, 0, 9)),
            'f1' : np.abs(angle(self._landmarks, 2, 7)),
            'f2' : np.abs(angle(self._landmarks, 4, 5)),
            'f3' : find_max(self._dists['L'], self._dists['M']),
            'f4' : slope(self._landmarks, 0, 9),
            'f5' : slope(self._landmarks, 2, 7),
            'f6' : slope(self._landmarks, 4, 5),
            # Eyes
            'f7' : np.abs(angle(self._landmarks, 10, 19)),
            'f8' : find_max(self._dists['Bl'], self._dists['Br']),
            'f9' : find_max(self._dists['D'],  self._dists['E']),
            'f10': find_max(self._dists['H'],  self._dists['I']),
            'f11': find_max_alt(self._dists['N'],  self._dists['O']),   # PROBLEM, 0 division
            'f12': find_max_alt(self._dists['Nl'], self._dists['Or']),  # PROBLEM
            'f13': find_max_alt(self._dists['Nr'], self._dists['Ol']),  # PROBLEM
            # Mouth
            'f14': np.abs(angle(self._landmarks, 28, 34)),
            'f15': find_max(self._dists['F'],  self._dists['G']),
            'f16': find_max(self._dists['Pl'], self._dists['Ql']),
            'f17': find_max(self._dists['Pu'], self._dists['Qu']),
            'f18': np.max((self._dists['Vl']/self._dists['A'] , self._dists['Vr']/self._dists['A'])),
            'f19': np.max((self._dists['Pl']/self._dists['W'] , self._dists['Ql']/self._dists['W'])),
            'f20': np.max((self._dists['Pu']/self._dists['W'] , self._dists['Qu']/self._dists['W'])),
            'f21': np.max((self._dists['Wl']/self._dists['W'] , self._dists['Wr']/self._dists['W'])),
            # Nose
            'f22': np.abs(angle(self._landmarks, 23, 27)),
            # Combined        
            'f23': 90-np.abs(angle(self._landmarks, 22, 37)),
            'f24': find_max(self._dists['J'], self._dists['K']),
            'f25': np.max((self._dists['T']/self._dists['A'] , self._dists['U']/self._dists['A'])),
            'f26': np.max((self._dists['R']/self._dists['A'] , self._dists['S']/self._dists['A'])),
            'f27': self._dists['C'] / self._dists['A'],
            'f28': self._dists['X'] / self._dists['A']
        }
        
    def features_to_arr(self):
        self._features_arr = np.array(list(self._features.values()),dtype=np.float32)
 
    def compute_pixel_scale(self):
        # averaging both eyes' iris rad in pixels (like rad of 5 pixels)
        pixel_iris_radius = (self._lefteye[2]+self._righteye[2])/2 
        # # physical dia (mm) / pixels = mm/pixel --> how many mm is one pixel measuring
        self._pixel_scale = self._physical_iris_diameter/(2*pixel_iris_radius)

    
    def scale_results(self):
        if self._scaled_by_pixel == False & self._scaled_by_proportion == False:
            self.compute_pixel_scale()

            for this_dist in self._dists:
                self._dists[this_dist] = self._dists[this_dist] * self._pixel_scale
            self.compute_features()

            self._scaled_by_pixel = True
            print("Distances and features have successfully scaled by pixel scale.")
            print("Pixel Scale = " + str(self._pixel_scale) + " (mm/pixel)")

        else: # Already scaled
            print("Distances and features already scaled. No action was performed.")
            print("Pixel Scale = " + str(self._pixel_scale) + " (mm/pixel)")

    
    def scale_by_bbox(self, bbox):
        
        if self._scaled_by_pixel == False & self._scaled_by_proportion == False:
            dy = bbox[3] - bbox[1]
            dx = bbox[2] - bbox[0]
            self._bbox_dist = np.sqrt(dy**2 + dx**2)

            for this_dist in self._dists:
                self._dists[this_dist] = self._dists[this_dist] / self._bbox_dist
            self.compute_features()

            self._scaled_by_proportion = True
            print("Distances and features have successfully scaled as proportion to Bbox distance.")
            print("Bbox distance (diagonal) = " + str(self._bbox_dist) + " (pixel)")
            
        else: # Already scaled
            print("Distances and features already scaled. No action was performed.")
            print("Bbox distance (diagonal) = " + str(self._bbox_dist) + " (pixel)")
    
    def save_dists(self, file_name="dists.csv"):
        csv_columns = ['dist', 'value'] 
        rows = list(self._dists.items())

        csv_file = file_name
        try:
            with open(csv_file, 'w') as f:
                write = csv.writer(f)
                write.writerow(csv_columns)
                write.writerows(rows)
        except IOError:
            print("Distances I/O error")


    def save_features(self, file_name="features.csv"):
        csv_columns = ['feature', 'value'] 
        rows = list(self._features.items())

        csv_file = file_name
        try:
            with open(csv_file, 'w') as f:
                write = csv.writer(f)
                write.writerow(csv_columns)
                write.writerows(rows)
        except IOError:
            print("Features I/O error")


    def save_results(self, dists_file_name="dists.csv", features_file_name="features.csv"):
        self.save_dists(dists_file_name)
        self.save_features(features_file_name)

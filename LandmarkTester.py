"""
Created Summer 2022

@author: Bingnan Huo (Nick) -- bh006 at bucknell.edu
"""

import numpy as np
import time
import multiprocess as mp
#import imgaug as ia

from GetLandmarks import GetLandmarks 

from landmark_utils import rotate_image, rotate_landmarks, to_gemma_landmarks
from matrix_utils import  arr_info
from sklearn.metrics import mean_squared_error

class LandmarkTester:
    def __init__(self, input_image, model_to_use="MEE", resolution=200, colored=False, gemma_lm=True):
        super().__init__()
       
        self._image = input_image
        self._model = model_to_use
        self._gemma_lm = gemma_lm

        start_time = time.time()
        self._og_landmarks = GetLandmarks(self._image, model_to_use, resolution, colored)._shape
        end_time = time.time()

        self._rot_center = (np.int16(0.5*input_image.shape[0]), 
                            np.int16(0.5*input_image.shape[1])) # Default rot center is the middle
        self._test_params = []
        self._test_params2 = []

        #self._aug_images = [] # No longer used, for better memory management
        self._test_landmarks = [] # shape: (num_tests, num_landmarks, 2)
        self._median_landmarks = []
        
        self._points_dist = [] # shape: (num_landmarks,num_tests, 2)
        self._xy_dist = [] # shape: (num_landmarks, 2, num_tests)
        
        self._RMSE = None
        
        #self._test_count = 0
        self._time_used = None

        print("===== Landmark Tester Initialized =====")
        print("Time used : {:.2f} sec".format(end_time - start_time))
        #print("Using model : {}".format(self._model))
        #print("Original landmarks has {} points".format(self._og_landmarks.shape[0]))
        
        print(arr_info(self._image))
        print(arr_info(self._og_landmarks))
        
        


#################################################################################
################################  Random Tests  #################################

    def test_random_rotation(self, rand_angle=10, num_test=20, rand_mode='uniform'):
        # Setting the distribution of the random angle
        if rand_mode == 'uniform':
            self._test_params = 2*(np.random.rand(num_test)-0.5)*rand_angle 
        elif rand_mode == 'normal': # if normal, set N(mu=0, sd=0.5*rand_angle); 95% should be within +- 2SD; so 95% within +-rand_angle
            self._test_params = np.random.normal(loc=0, scale=rand_angle/2, size=num_test) 
        else:
            raise ValueError("Random mode must be 'uniform' or 'normal'.")
        self.rotations()

    def test_fixed_rotation(self, test_angle=10, num_steps=20):
        # Setting the distribution of the stepped angle
        steps = 2*test_angle / num_steps
        self._test_params = np.arange(-test_angle, test_angle+steps, steps, dtype=np.half)
        self.rotations()

    def test_no_rotation(self, num_test=10):
        self._test_params = np.zeros(num_test)
        self.rotations()
        
        
    # Test intensity shifts
    def test_intensities(self, scale=20, num_test=20):
        # Setting the distribution of the random intensities to add to the image
        self._test_params = np.random.randint(-scale, scale, size=num_test)
        self.noises()
    

    def test_noises(self, scale=20, dist='uniform', num_test=20, per_channel=True):
        # Setting the distribution of the random intensities to add to the image
        if dist == 'uniform':
            if per_channel is True:
                self._test_params = (2*(np.random.rand(*((num_test,)+self._image.shape))-0.5)*scale)
            else:
                one_chan = 2*(np.random.rand(*((num_test,)+self._image.shape[:2]))-0.5)*scale
                self._test_params = np.stack([one_chan, one_chan, one_chan], axis=-1)
                
        elif dist == 'normal' or dist == 'laplace' or dist == 'poisson':
            if per_channel is True:
                self._test_params = np.random.normal(loc=0, scale=scale, size=(num_test,)+self._image.shape)
            else:
                one_chan = (np.random.normal(loc=0, scale=scale, size=(num_test,)+self._image.shape[:2]))
                self._test_params = np.stack([one_chan, one_chan, one_chan], axis=-1)
                
        else:
            raise TypeError("Error: dist must be 'uniform', 'normal', 'laplace', or 'poisson'.")
            
        self._test_params = np.int16(self._test_params)
        self.noises()

    
    def test_combined(self, rand_angle=7, scale=25, num_test=30, rand_mode='normal', per_channel=True):
        # Setting the distribution of the random angle
        if rand_mode == 'uniform':
            self._test_params = 2*(np.random.rand(num_test)-0.5)*rand_angle 
        elif rand_mode == 'normal': # if normal, set N(mu=0, sd=0.5*rand_angle); 95% should be within +- 2SD; so 95% within +-rand_angle
            self._test_params = np.random.normal(loc=0, scale=rand_angle/2, size=num_test) 
        else:
            raise ValueError("Random mode must be 'uniform' or 'normal'.")
        
        dist = rand_mode
        # Setting the distribution of the random intensities to add to the image
        if dist == 'uniform':
            if per_channel is True:
                self._test_params2 = (2*(np.random.rand(*((num_test,)+self._image.shape))-0.5)*scale)
            else:
                one_chan = 2*(np.random.rand(*((num_test,)+self._image.shape[:2]))-0.5)*scale
                self._test_params2 = np.stack([one_chan, one_chan, one_chan], axis=-1)
                
        elif dist == 'normal' or dist == 'laplace' or dist == 'poisson':
            if per_channel is True:
                self._test_params2 = np.random.normal(loc=0, scale=scale, size=(num_test,)+self._image.shape)
            else:
                one_chan = (np.random.normal(loc=0, scale=scale, size=(num_test,)+self._image.shape[:2]))
                self._test_params2 = np.stack([one_chan, one_chan, one_chan], axis=-1)
        else:
            raise TypeError("Error: dist must be 'uniform', 'normal', 'laplace', or 'poisson'.")
        
        self._test_params2 = np.int16(self._test_params2)
        self._test_params2 = np.array(list(zip(self._test_params, self._test_params2)), dtype='object')
                   
        self.combined_augs()
         
        
#################################################################################
###################################  Tools  #####################################


    def rotations(self): # Now with multiprocess
        print("Testing {} rotations...".format(self._test_params.size))
        print("First 10 Angles:")
        print(self._test_params[:10])
        self.run_tests_mp(self.single_rotation)
        
    def single_rotation(self, angle):
        rotated = rotate_image(self._image, self._rot_center, angle)
        rotated_landmarks = GetLandmarks(rotated, self._model)._shape
        
        #print("Roughly {} tests have completed!".format(self._test_count))
        print("One test has completed...")
        res_landmarks = rotate_landmarks(rotated_landmarks, self._rot_center, -angle)
        if self._gemma_lm is True:
            return to_gemma_landmarks(res_landmarks)
        return res_landmarks
    
    
    # Changes the intensity of one image
    def noises(self):
        print("Testing {} noises / intensity shifts...".format(self._test_params.shape[0]))
        print("First Intensities:")
        print(self._test_params[0])
        self.run_tests_mp(self.noise)    
    
       
    # Add noise to an image
    def noise(self, added_noise):
        # noise can be an array (actual noise) or a int (total intensity shift)
        new_img = (np.clip(np.int16(self._image) + added_noise, 0, 255)).astype(np.uint8)
        print("One test has completed...")
        res_landmarks = GetLandmarks(new_img, self._model)._shape
        if self._gemma_lm is True:
            return to_gemma_landmarks(res_landmarks)
        return res_landmarks
        
    
    def combined_augs(self):
        print("Testing {} rotations & noises...".format(self._test_params2.shape[0]))
        #print("First param pairs:")
        #print(self._test_params2[0])
        self.combined_tests_mp(self.combined_aug) 
        
    def combined_aug(self, angle, added_noise):
        # add noise, then rotate
        new_img = (np.clip(np.int16(self._image) + added_noise, 0, 255)).astype(np.uint8)
        rotated = rotate_image(new_img, self._rot_center, angle)
        #self._aug_images.append(rotated)

        rot_landmarks = GetLandmarks(rotated, self._model)._shape
        res_landmarks = rotate_landmarks(rot_landmarks, self._rot_center, -angle)
        print("One test has completed...")
        if self._gemma_lm is True:
            return to_gemma_landmarks(res_landmarks)
        return res_landmarks
        
        
    # Run tests with supplied function in multiprocessing
    def run_tests_mp(self, func):
        print("===== Tests Started! =====")
        start_time = time.time()
        
        #with mp.get_context("spawn").Pool(processes=19, maxtasksperchild=15) as pool:
        with mp.get_context("spawn").Pool() as pool:
            self._test_landmarks = pool.map(func, self._test_params)

        #self._points_dist = np.transpose(self._test_landmarks, (1, 0, 2))
        self._xy_dist = np.transpose(self._test_landmarks, (1, 2, 0))
        self._median_landmarks = np.median(self._xy_dist, axis=2)
        
        end_time = time.time()

        print("===== Tests Completed! =====")
        print("Total Time : {:.2f} sec".format(end_time - start_time))
        print("Average time/test : {:.2f} sec per test\n".format((end_time - start_time)/self._test_params.shape[0]))
        print()
        
    # Run tests with supplied function in multiprocessing
    def combined_tests_mp(self, func):
        print("===== Tests Started! =====")
        start_time = time.time()
        
        with mp.get_context("spawn").Pool() as pool:
            self._test_landmarks = pool.starmap(func, self._test_params2)

        #self._points_dist = np.transpose(self._test_landmarks, (1, 0, 2))
        self._xy_dist = np.transpose(self._test_landmarks, (1, 2, 0))
        self._median_landmarks = np.median(self._xy_dist, axis=2)
        
        end_time = time.time()

        print("===== Tests Completed! =====")
        print("Total Time : {:.2f} sec".format(end_time - start_time))
        print("Average time/test : {:.2f} sec per test\n".format((end_time - start_time)/self._test_params2.shape[0]))
        print()

    
    def RMSE_all(self):
        # Calculate the RMSE of all the landmarks
        self._RMSE = []
        for i in range(self._test_landmarks.shape[0]):
            self._RMSE.append(mean_squared_error(self._og_landmarks, self._test_landmarks[i], squared=False))
        print("RMSE of all landmarks:")
        print(self._RMSE[:10])
        print("Average RMSE: {:.2f}".format(np.mean(self._RMSE)))
        print("Standard Deviation of RMSE: {:.2f}".format(np.std(self._RMSE)))
        print()
        return self._RMSE
    
    
    
#################################################################################
##############################  Deprecated Tools  ###############################


       


    '''    
    def noises(self, scale=10, dist='uniform', per_channel=1, num_test=20):
        # Setting the distribution of the random intensities to add to the image
        if dist == 'uniform':
            aug = ia.AddElementwise((-scale, scale), per_channel=per_channel)
        elif dist == 'normal' or 'gaussian':
            aug = ia.AdditiveGaussianNoise((-scale, scale), per_channel=per_channel)
        elif dist == 'laplace':
            aug = ia.AdditiveLaplaceNoise((-scale, scale), per_channel=per_channel)
        elif dist == 'poisson':
            aug = ia.AdditivePoissonNoise((-scale, scale), per_channel=per_channel)
            
        
        start_time = time.time()
        print("Testing {} noises...".format(num_test))
        #print("Intensities:")
        #print(self._test_intensities)
        with ia.multicore.Pool(aug, processes=-1, maxtasksperchild=20, seed=1) as pool:
            batches_aug = pool.map_batches(batches)

        ia.imshow(batches_aug[0].images_aug[0])

        with mp.get_context("spawn").Pool() as pool:
            self._test_landmarks = pool.starmap(aug, self._test_params)

        end_time = time.time()
        print(self._test_landmarks)
        self._points_dist = np.transpose(self._test_landmarks, (1, 0, 2))
        
        print("===== Tests Completed! =====")
        print("Total Time : {:.2f} sec".format(end_time - start_time))
        print("Average time/test : {:.2f} sec per test\n".format((end_time - start_time)/self._test_params.size))
    '''


    '''
    def rotations_sp(self):
        rot_center = self._rot_center
        angles = self._test_angles
        num_test = angles.shape[0]

        # For each angle, rotate the original image and find landmarks,
        # then rotate landmarks back and store them
        num_points = self._og_landmarks.shape[0]
        ###rotated_images = np.zeros((num_test,)+self._image.shape, dtype=np.uint8)
        test_landmarks = np.zeros((num_test,num_points,2), dtype=np.uint16)
        time_used = np.zeros((num_test), dtype=np.half)
        
        for i in range(num_test):
            start_time = time.time()
            print("TEST {}/{}".format(i+1,num_test))

            rotated = rotate_image(self._image, rot_center, angles[i])
            ###rotated_images[i] = rotated
            rotated_landmarks = GetLandmarks(rotated, self._model)._shape
            test_landmarks[i] = rotate_landmarks(rotated_landmarks, 
                                                        rot_center,
                                                        -1*angles[i])

            end_time = time.time()
            print ("--time used: {:.2f} sec\n".format(end_time - start_time))
            time_used[i] = end_time - start_time

        # Update object stored images and landmarks
        ###self._rotated_images = rotated_images
        self._test_landmarks = test_landmarks
        self._time_used = time_used
        
        # Arrange the landmarks by specific points
        self._points_dist = np.transpose(test_landmarks, (1, 0, 2))
        


    def test_mp(self, rot_center, test_angle=10, num_steps=20):
        self._rot_center = rot_center
        steps = 2*test_angle / num_steps
        self._test_angles = np.arange(-test_angle, test_angle+steps, steps, dtype=np.half)
        self.rotations_mp()
    '''



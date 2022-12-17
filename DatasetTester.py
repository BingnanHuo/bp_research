"""
Created Summer 2022

@author: Bingnan Huo (Nick) -- bh006 at bucknell.edu
"""

import csv
from pathlib import Path

import cv2
import multiprocess as mp
import numpy as np
import pandas as pd

from landmark_utils import rotate_image, rotate_landmarks
from matrix_utils import arr_info, normalize_ubyte, preprocess_img

#from Emotrics_GetLandmarks_new import GetLandmarks as GetLandmarks_new
from sklearn.metrics import mean_squared_error

class DatasetIntegrator:
    def __init__(self, dataset_path, model_to_use="MEE"):
        super().__init__()
        self._dataset_path = Path(dataset_path)

        self._ALS_path = self._dataset_path / "ALS"
        self._stroke_path = self._dataset_path / "Stroke"
        self._control_path = self._dataset_path / "Healthy controls"

        self._ALS_patients = DatasetTester(self._ALS_path, model_to_use)._patients
        self._stroke_patients = DatasetTester(self._stroke_path, model_to_use)._patients
        self._control_patients = DatasetTester(self._control_path, model_to_use)._patients
        
        print("DatasetIntegrator initialized.")
        print("ALS:", len(self._ALS_patients))
        print("Stroke:", len(self._stroke_patients))
        print("Control:", len(self._control_patients))

    def patients_generator(self, disease_type=None, tasks=None, batch_size=1):
        """
        Generator for the patients.
        Yields a batch of patients.
        """
        disease_type = disease_type.lower()
        # If disease_type is None, yield all patients
        # If disease_type is not None, only yield patients with the given disease_type
        if disease_type is None or disease_type == "all":
            patients = self._ALS_patients | self._stroke_patients | self._control_patients
        elif disease_type == "als":
            patients = self._ALS_patients
        elif disease_type == "stroke":
            patients = self._stroke_patients
        elif disease_type == "control":
            patients = self._control_patients
        elif disease_type == "als_stroke" or disease_type == "stroke_als" or disease_type == "both":
            patients = self._ALS_patients | self._stroke_patients
        else:
            raise ValueError("Disease type not recognized.")


        # Now process task_type; should be either None or a tuple of (task_type1, task_type2, ...) 
        all_tasks = ('BBP_NORMAL', 'DDK_PATAKA', 'DDK_PA', 'NSM_BLOW', 'NSM_KISS', 'NSM_OPEN', 'NSM_SPREAD', 'NSM_BIGSMILE', 'NSM_BROW')
        if tasks is None or tasks == "all":
            tasks = all_tasks
        elif isinstance(tasks, str) and tasks in all_tasks: # one single task
            tasks = (tasks)
        else:
            tasks = tuple(tasks)
            
        # Now yield patients with the given tasks
        tasks_to_drop = tuple([t for t in all_tasks if t not in tasks])
        for p in patients:
            for t in tasks_to_drop:
                patients[p].pop(t, None)
                
        # Now needed patients and tasks are good in the dict; yield patients in batches        

        n_patients = len(list(patients.keys()))

        # batch size functionality, default is just 1 by 1. 
        for i in range(0, n_patients, batch_size):
            yield np.array(list(patients.values()))[i:i+batch_size]




##############################################################################################################################
##############################################################################################################################
##############################################################################################################################

class DatasetTester:
    def __init__(self, dataset_path, model_to_use="MEE"):
        super().__init__()
        self._dataset_path = Path(dataset_path)

        self._landmarks_path = self._dataset_path / 'Landmarks_gt'
        self._bbox_path = self._dataset_path / 'Bbox_gt'
        self._images_path = self._dataset_path / 'Frames'
        
        # dict containing all the patients, in each patient there is a dict with the tasks
        self._patients = {} #  for each task there is a dict with the frames, then for each frame there is a dict with the landmarks, bboxes, images

        self.load_landmarks_bbox_images()
        
    
    def load_landmarks_bbox_images(self):
        """ read in csv files of landmarks and bboxes
            for a single patient, there are 7 different tasks / poses
            for each task, there are multiple images (frames listed in the csv file)
        """   
        lm_filenames = sorted([f.name for f in self._landmarks_path.iterdir() if f.is_file()]) # landmarks filenames, same for bboxes
        patients_id = [f[0:4] for f in lm_filenames] # get first 4 characters of filename
        patients = {p:{} for p in patients_id}
        
        for p in patients:  
            print('Patient', p, 'loading...') 
            # Every patient has a dictionary of tasks, each task has a dict of frames, which contains dict of landmarks and bboxes
            tasks = ['_'.join(t.split('_')[2:4]) for t in lm_filenames if t[0:4]==p] # list of task names for patient p
            patients[p] = {t:{} for t in tasks} # create a dict for each task
            
            for t in tasks:
                # for every task, load the landmarks and bboxes
                this_filename = '_'.join((p,'02',t,'color.txt'))
                cur_lm_path = self._landmarks_path / this_filename
                cur_bbox_path = self._bbox_path / this_filename
                
                frames, landmarks = self.load_landmarks(cur_lm_path)
                _, bboxes = self.load_bboxes(cur_bbox_path)

                patients[p][t] = {f:{} for f in frames} # now we have a dict of frames

                # need to fill every frame's dict with landmarks and bboxes, and also have a spot for image
                for f in frames:
                    idx = np.where(frames==f)[0][0]
                    img_path = self._images_path / '_'.join((p,'02',t,'color.avi',str(f)+'.jpg'))

                    patients[p][t][f]['landmarks_gt'] = landmarks[idx]
                    patients[p][t][f]['bbox'] = bboxes[idx]
                    patients[p][t][f]['image'] = cv2.imread(str(img_path))
                    patients[p][t][f]['landmarks_pred'] = None
        
        self._patients = patients
        

        


#################################################################################
###################################  Tools  #####################################
    def load_landmarks(self, lm_path):
        """
        Loads the landmarks from the given path.
        
        lm_path: Path object to the landmarks file
        
        Return: a dict - key: frame number, value: np.array of landmarks (68,2)
        """
        with open(lm_path) as f:
            next(f) # skip the first line
            reader = csv.reader(f, delimiter=',')
            lines = np.array(list(reader))
            
        frames = lines[:,0].astype(np.uint16)
        landmarks = lines[:,1:].reshape(-1,68,2).astype(np.float32)     # (n_frames, n_landmarks, 2)
        return frames, landmarks



    def load_bboxes(self, bbox_path):
        """
        Loads the bounding boxes from the given path.
        
        bbox_path: Path object to the bounding boxes file
        
        Return: a dict - key: frame number, value: np.array of bounding boxes (n_bboxes, 4)
        """
        with open(bbox_path) as f:
            next(f) # skip the first line
            reader = csv.reader(f, delimiter=',')
            lines = np.array(list(reader))

        frames = lines[:,0].astype(np.uint16)
        bboxes = lines[:,1:].reshape(-1,4).astype(np.float32)     # (n_frames, bboxes' 4 coords, 4) [x1,y1,x2,y2]
        return frames, bboxes
    
    

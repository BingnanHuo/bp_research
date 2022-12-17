import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from GetLandmarks import GetLandmarks

def to_gemma_landmarks(input_landmarks):
    #function to change from 61 pts to 51 pts gemma model
    if input_landmarks.shape == (51,2):
        return input_landmarks
    
    gemma_landmarks = np.zeros((51,2),dtype=int)
    gemma_landmarks[0] = input_landmarks[17]
    gemma_landmarks[1] = input_landmarks[18]
    gemma_landmarks[2] = input_landmarks[19]
    gemma_landmarks[3] = input_landmarks[20]
    gemma_landmarks[4] = input_landmarks[21]
    gemma_landmarks[5] = input_landmarks[22]
    gemma_landmarks[6] = input_landmarks[23]
    gemma_landmarks[7] = input_landmarks[24]
    gemma_landmarks[8] = input_landmarks[25]
    gemma_landmarks[9] = input_landmarks[26]
    gemma_landmarks[10] = input_landmarks[36]
    gemma_landmarks[11] = input_landmarks[37]
    gemma_landmarks[12] = input_landmarks[38]
    gemma_landmarks[13] = input_landmarks[39]
    gemma_landmarks[14] = input_landmarks[40]
    gemma_landmarks[15] = input_landmarks[41]
    gemma_landmarks[16] = input_landmarks[42]
    gemma_landmarks[17] = input_landmarks[43]
    gemma_landmarks[18] = input_landmarks[44]
    gemma_landmarks[19] = input_landmarks[45]
    gemma_landmarks[20] = input_landmarks[46]
    gemma_landmarks[21] = input_landmarks[47]
    gemma_landmarks[22] = input_landmarks[30]
    gemma_landmarks[23] = input_landmarks[31]
    gemma_landmarks[24] = input_landmarks[32]
    gemma_landmarks[25] = input_landmarks[33]
    gemma_landmarks[26] = input_landmarks[34]
    gemma_landmarks[27] = input_landmarks[35]
    gemma_landmarks[28] = input_landmarks[48]
    gemma_landmarks[29] = input_landmarks[49]
    gemma_landmarks[30] = input_landmarks[50]
    gemma_landmarks[31] = input_landmarks[51]
    gemma_landmarks[32] = input_landmarks[52]
    gemma_landmarks[33] = input_landmarks[53]
    gemma_landmarks[34] = input_landmarks[54]
    gemma_landmarks[35] = input_landmarks[55]
    gemma_landmarks[36] = input_landmarks[56]
    gemma_landmarks[37] = input_landmarks[57]
    gemma_landmarks[38] = input_landmarks[58]
    gemma_landmarks[39] = input_landmarks[59]
    gemma_landmarks[40] = input_landmarks[60]
    gemma_landmarks[41] = input_landmarks[61]
    gemma_landmarks[42] = input_landmarks[62]
    gemma_landmarks[43] = input_landmarks[63]
    gemma_landmarks[44] = input_landmarks[64]
    gemma_landmarks[45] = input_landmarks[65]
    gemma_landmarks[46] = input_landmarks[66]
    gemma_landmarks[47] = input_landmarks[67]
    gemma_landmarks[48] = input_landmarks[0]
    gemma_landmarks[49] = input_landmarks[16]
    gemma_landmarks[50] = input_landmarks[8]
    return gemma_landmarks



def rotate_image(im, rot_center, rot_angle,scale=1):
    # load the image you want by certain degree at some center
    # output will also be an image (opencv image).
    rot_mat = cv2.getRotationMatrix2D(center=rot_center, angle=rot_angle, scale=scale)
    rotated = cv2.warpAffine(im, rot_mat, (im.shape[1], im.shape[0]))
    return rotated


def rotate_landmarks(arr, rot_center, rot_angle, scale=1):
    # make sure that coords are integers
    rot_center = tuple(np.int16(rot_center))
    rot_mat = cv2.getRotationMatrix2D(center=rot_center, angle=rot_angle, scale=scale)
    arr_pad = np.pad(np.transpose(arr), 
                        ((0,1),(0,0)), 
                        'constant', 
                        constant_values=(0,1))
    arr_rotated = np.float32(np.transpose(np.matmul(rot_mat, arr_pad))) 
    return arr_rotated

# Display landmarks on top of patient image
def vis_landmarks(im, og_landmarks=None, which_points=np.arange(68), figsize=(4,3), title='Image', show_ticks = True, **kwargs):
    if og_landmarks is None:
        og_landmarks = GetLandmarks(im)._shape
    
    f, ax = plt.subplots(1,1, figsize=figsize)
    ax.set_title(title)

    ax.imshow(im, **kwargs)
    og_points = np.transpose(og_landmarks)
    ax.scatter(x=og_points[0], y=og_points[1], c='#42ff55', s=20, alpha=1, edgecolor='none')
    
    if not show_ticks:
        ax.get_xaxis().set_visible(False);
        ax.get_yaxis().set_visible(False);
            
    plt.tight_layout()
    
    
# Get random hex color
def random_color():
    return '#{:06x}'.format(np.random.randint(0, 0xFFFFFF))
    
    
# Display landmarks on top of patient image
def compare_landmarks(im, lm0, lm1, other_lms=None, figsize=(4,3), title='Image', show_ticks = True, **kwargs): 
    f, ax = plt.subplots(1,1, figsize=figsize)
    ax.set_title(title)

    ax.imshow(im, **kwargs)
    lm0 = np.transpose(lm0)
    lm1 = np.transpose(lm1)
    
    ax.scatter(x=lm0[0], y=lm0[1], c='#42ff55', s=20, alpha=1, edgecolor='none')
    ax.scatter(x=lm1[0], y=lm1[1], c='#f2ff42', s=20, alpha=1, edgecolor='none')

    if other_lms is not None:
        for lm in other_lms:
            lm = np.transpose(lm)
            ax.scatter(x=lm[0], y=lm[1], c=random_color(), s=20, alpha=1, edgecolor='none')
    
    if not show_ticks:
        ax.get_xaxis().set_visible(False);
        ax.get_yaxis().set_visible(False);
            
    plt.tight_layout()




#Calculate error between og_landmarks and predicted_landmarks
   
def RMSE(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, multioutput='uniform_average', squared=False)



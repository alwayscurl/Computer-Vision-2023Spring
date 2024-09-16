import numpy as np
import cv2

class Difference_of_Gaussian(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.sigma = 2**(1/4)
        self.num_octaves = 2
        self.num_DoG_images_per_octave = 4
        self.num_guassian_images_per_octave = self.num_DoG_images_per_octave + 1
        self.dog_images = []

    def get_keypoints(self, image):
        ### TODO ####
        # Step 1: Filter images with different sigma values (5 images per octave, 2 octave in total)
        # - Function: cv2.GaussianBlur (kernel = (0, 0), sigma = self.sigma**___)
        gaussian_images = [image]
        for i in range(4):
            sigma = self.sigma**(i+1)
            gaussian_images.append(cv2.GaussianBlur(image, (0, 0), sigma))
        
        down_sample_image = cv2.resize(gaussian_images[-1], (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
        gaussian_images_down = [down_sample_image]
        for i in range(4):
            sigma = self.sigma**(i+1)
            gaussian_images_down.append(cv2.GaussianBlur(down_sample_image, (0, 0), sigma))
        
        # Step 2: Subtract 2 neighbor images to get DoG images (4 images per octave, 2 octave in total)
        # - Function: cv2.subtract(second_image, first_image)
        dog_images = []
        for i in range(self.num_DoG_images_per_octave):
            dog_images.append(cv2.subtract(gaussian_images[i+1], gaussian_images[i]))
            self.dog_images.append(cv2.subtract(gaussian_images[i+1], gaussian_images[i]))
        dog_images_down = []
        for i in range(self.num_DoG_images_per_octave):
            dog_images_down.append(cv2.subtract(gaussian_images_down[i+1], gaussian_images_down[i]))
            self.dog_images.append(cv2.subtract(gaussian_images_down[i+1], gaussian_images_down[i]))
            

        # Step 3: Thresholding the value and Find local extremum (local maximun and local minimum)
        #         Keep local extremum as a keypoint
        keypoints = self.extremum_points(dog_images, dog_images[0].shape, is_down=False)
        keypoints = np.append(keypoints, self.extremum_points(dog_images_down, dog_images_down[0].shape, is_down=True), axis=0)
        
        # Step 4: Delete duplicate keypoints
        # - Function: np.unique
        keypoints = np.unique(keypoints, axis=0)
        # print(keypoints.shape)
        # sort 2d-point by y, then by x
        keypoints = keypoints[np.lexsort((keypoints[:,1],keypoints[:,0]))]
        return keypoints
    
    def extremum_points(self, dog_images, shape, is_down=False):
        keypoints = np.zeros((0, 2))
        for k in range(2):
            for i in range(1, shape[0]-1):
                for j in range(1, shape[1]-1):
                    is_max = True
                    is_min = True
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dog_images[k+1][i][j] >= dog_images[k][i+dx][j+dy] or dog_images[k+1][i][j] >= dog_images[k+2][i+dx][j+dy]:
                                is_min = False
                            if dog_images[k+1][i][j] <= dog_images[k][i+dx][j+dy] or dog_images[k+1][i][j] <= dog_images[k+2][i+dx][j+dy]:
                                is_max = False
                            if dx != 0 or dy != 0:
                                if dog_images[k+1][i][j] >= dog_images[k+1][i+dx][j+dy]:
                                    is_min = False
                                if dog_images[k+1][i][j] <= dog_images[k+1][i+dx][j+dy]:
                                    is_max = False
                    if (is_max or is_min) and abs(dog_images[k+1][i][j]) > self.threshold:
                        # print('find keypoint')
                        if is_down:
                            keypoints = np.append(keypoints, [[2*i, 2*j]], axis=0)
                        else:
                            keypoints = np.append(keypoints, [[i, j]], axis=0)
        # print(keypoints.shape)
        keypoints = keypoints.astype(int)
        return keypoints
    
    def get_dogimages(self):
        return self.dog_images
                
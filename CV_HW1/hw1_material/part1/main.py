import numpy as np
import cv2
import argparse
from DoG import Difference_of_Gaussian


def plot_keypoints(img_gray, keypoints, save_path):
    img = np.repeat(np.expand_dims(img_gray, axis = 2), 3, axis = 2)
    for y, x in keypoints:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
    cv2.imwrite(save_path, img)

def main():
    parser = argparse.ArgumentParser(description='main function of Difference of Gaussian')
    parser.add_argument('--threshold', default=5.0, type=float, help='threshold value for feature selection')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    args = parser.parse_args()

    print('Processing %s ...'%args.image_path)
    img = cv2.imread(args.image_path, 0).astype(np.float32)

    ### TODO ###
    Dog = Difference_of_Gaussian(args.threshold)
    keypoints = Dog.get_keypoints(img)
    dog_images = Dog.get_dogimages()
    for i in range(len(dog_images)):
        min = np.min(dog_images[i])
        max = np.max(dog_images[i])
        for x in range(dog_images[i].shape[0]):
            for y in range(dog_images[i].shape[1]):
                dog_images[i][x][y] = (dog_images[i][x][y] - min) / (max - min) * 255
    
    for i in range(len(dog_images)):
       cv2.imwrite('./output/DoG%d-%d.png'%(i/4+1, i%4+1), dog_images[i])
    
    # Dog2_thr1 = Difference_of_Gaussian(1.0)
    # keypoints_2_threshold1 = Dog2_thr1.get_keypoints(img)
    # plot_keypoints(img, keypoints_2_threshold1, './output/DoG_thr1.png')
    # Dog2_thr2 = Difference_of_Gaussian(2.0)
    # keypoints_2_threshold2 = Dog2_thr2.get_keypoints(img)
    # plot_keypoints(img, keypoints_2_threshold2, './output/DoG_thr2.png')
    # Dog2_thr3 = Difference_of_Gaussian(3.0)
    # keypoints_2_threshold3 = Dog2_thr3.get_keypoints(img)
    # plot_keypoints(img, keypoints_2_threshold3, './output/DoG_thr3.png')

if __name__ == '__main__':
    main()
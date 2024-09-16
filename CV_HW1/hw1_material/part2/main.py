import numpy as np
import cv2
import argparse
import os
from JBF import Joint_bilateral_filter


def main():
    parser = argparse.ArgumentParser(description='main function of joint bilateral filter')
    parser.add_argument('--image_path', default='./testdata/1.png', help='path to input image')
    parser.add_argument('--setting_path', default='./testdata/1_setting.txt', help='path to setting file')
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ### TODO ###
    f = open(args.setting_path, 'r')
    text = f.readlines()
    text = [i.strip() for i in text]        
    weight  = []
    for i in range(5):
        weight.append(text[i+1].split(','))
        for j in range(3):
            weight[i][j] = float(weight[i][j])
    sigma_s = int(text[6].split(',')[1])
    sigma_r = float(text[6].split(',')[3])
    f.close()
    
    JBF = Joint_bilateral_filter(sigma_s, sigma_r)
    gray_scale = []
    for i in range(5):
        gray_scale.append(img_rgb[:,:,0]*weight[i][0] + img_rgb[:,:,1]*weight[i][1] + img_rgb[:,:,2]*weight[i][2])
    gray_scale.append(img_gray)
    
    BF_image = JBF.joint_bilateral_filter(img_rgb, img_rgb)
    
    JBF_image = []
    cost = []
    for i in range(6):
        JBF_image.append(JBF.joint_bilateral_filter(img_rgb, gray_scale[i].astype(np.uint8)))
        cost.append(np.sum(np.abs(JBF_image[i].astype('int32')-BF_image.astype('int32'))))
    print(cost)
    cost.pop()
    min_cost = min(cost)
    min_index = cost.index(min_cost)
    max_cost = max(cost)
    max_index = cost.index(max_cost)
    print(min_index, max_index)
    # original rgb image
    cv2.imwrite('./output/original_2.png', img)
    # highest cost rgb image and guidance image
    cv2.imwrite('./output/highest_cost_2.png', cv2.cvtColor(JBF_image[max_index], cv2.COLOR_RGB2BGR))
    cv2.imwrite('./output/highest_cost_guidance_2.png', gray_scale[max_index])
    # lowest cost rgb image and guidance image
    cv2.imwrite('./output/lowest_cost_2.png', cv2.cvtColor(JBF_image[min_index], cv2.COLOR_RGB2BGR))
    cv2.imwrite('./output/lowest_cost_guidance_2.png', gray_scale[min_index])

    
        
    
    
    


if __name__ == '__main__':
    main()
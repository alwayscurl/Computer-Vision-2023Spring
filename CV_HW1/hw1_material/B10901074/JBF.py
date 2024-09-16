import numpy as np
import cv2


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        ### TODO ###
        range_kernel_table = np.zeros((256, 256))
        for i in range(256):
            for j in range(256):
                range_kernel_table[i][j] = np.exp(-(((i-j)/255)**2)/(2*self.sigma_r**2))
                # i, j: pixel value of guidance image
                
        spatial_kernel_table = np.zeros((self.wndw_size, self.wndw_size))
        for i in range(self.wndw_size):
            for j in range(self.wndw_size):
                spatial_kernel_table[i][j] = np.exp(-((i-self.pad_w)**2+(j-self.pad_w)**2)/(2*self.sigma_s**2))
                # i, j: difference of pixel position relative to center pixel
        
        if len(guidance.shape) == 2:
            bilateral_filter = np.zeros((self.wndw_size, self.wndw_size, img.shape[0], img.shape[1]))
            bilateral_image = np.zeros((self.wndw_size, self.wndw_size, img.shape[0], img.shape[1], img.shape[2]))
            for i in range(-self.pad_w, self.pad_w+1):
                for j in range(-self.pad_w, self.pad_w+1):
                    bilateral_filter[i+self.pad_w][j+self.pad_w] = spatial_kernel_table[i+self.pad_w][j+self.pad_w] * range_kernel_table[guidance, padded_guidance[self.pad_w+i:self.pad_w+i+img.shape[0], self.pad_w+j:self.pad_w+j+img.shape[1]]]
                    bilateral_image[i+self.pad_w][j+self.pad_w] = padded_img[i+self.pad_w:i+self.pad_w+img.shape[0], j+self.pad_w:j+self.pad_w+img.shape[1]] * np.reshape(bilateral_filter[i+self.pad_w][j+self.pad_w], (img.shape[0], img.shape[1], 1))
            output = np.sum(bilateral_image, axis=(0, 1))/np.sum(np.reshape(bilateral_filter, (*bilateral_filter.shape, 1)), axis=(0, 1))
        elif len(guidance.shape) == 3:
            bilateral_filter = np.zeros((self.wndw_size, self.wndw_size, img.shape[0], img.shape[1]))
            bilateral_filter_3d = np.zeros((self.wndw_size, self.wndw_size, img.shape[0], img.shape[1], img.shape[2]))
            bilateral_image = np.zeros((self.wndw_size, self.wndw_size, img.shape[0], img.shape[1], img.shape[2]))
            for i in range(-self.pad_w, self.pad_w+1):
                for j in range(-self.pad_w, self.pad_w+1):
                    bilateral_filter_3d[i+self.pad_w][j+self.pad_w] = range_kernel_table[guidance, padded_guidance[self.pad_w+i:self.pad_w+i+img.shape[0], self.pad_w+j:self.pad_w+j+img.shape[1]]]
                    bilateral_filter[i+self.pad_w][j+self.pad_w] = spatial_kernel_table[i+self.pad_w][j+self.pad_w] * bilateral_filter_3d[i+self.pad_w][j+self.pad_w][:,:,0] * bilateral_filter_3d[i+self.pad_w][j+self.pad_w][:,:,1] * bilateral_filter_3d[i+self.pad_w][j+self.pad_w][:,:,2]
                    bilateral_image[i+self.pad_w][j+self.pad_w] = padded_img[i+self.pad_w:i+self.pad_w+img.shape[0],j+self.pad_w:j+self.pad_w+img.shape[1]] * np.reshape(bilateral_filter[i+self.pad_w][j+self.pad_w], (img.shape[0], img.shape[1], 1))
            output = np.sum(bilateral_image, axis=(0, 1))/np.sum(np.reshape(bilateral_filter, (*bilateral_filter.shape, 1)) ,axis=(0, 1))

        
        return np.clip(output, 0, 255).astype(np.uint8)

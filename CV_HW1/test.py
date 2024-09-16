import numpy as np

if __name__ == '__main__':
    pad_w = 3
    wndw_size = 3*2+1
    sigma_r = 1.0
    range_kernel_table = np.zeros((256, 256))
    for i in range(256):
        for j in range(256):
            range_kernel_table[i][j] = a[[1,2]:[3,4]]
    img_1 = np.random.randint(0, 256, (5, 5))
    img_2 = np.random.randint(0, 256, (11, 11))
    range_kernel = np.zeros((wndw_size, wndw_size, img_1.shape[0], img_1.shape[1]))
    
    print(range_kernel_table[[]][[]])    
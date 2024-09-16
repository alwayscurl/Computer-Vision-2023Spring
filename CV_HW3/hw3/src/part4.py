import numpy as np
import cv2
import random
from tqdm import tqdm
from utils import solve_homography, warping

random.seed(999)

def panorama(imgs):
    """
    Image stitching with estimated homograpy between consecutive
    :param imgs: list of images to be stitched
    :return: stitched panorama
    """
    h_max = max([x.shape[0] for x in imgs])
    w_max = sum([x.shape[1] for x in imgs])

    # create the final stitched canvas
    dst = np.zeros((h_max, w_max, imgs[0].shape[2]), dtype=np.uint8)
    dst[:imgs[0].shape[0], :imgs[0].shape[1]] = imgs[0]
    last_best_H = np.eye(3)
    out = None

    # for all images to be stitched:
    for idx in tqdm(range(len(imgs)-1)):
        im1 = imgs[idx]
        im2 = imgs[idx + 1]

        # TODO: 1. feature detection & matching
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(im1, None)
        kp2, des2 = orb.detectAndCompute(im2, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        # TODO: 2. apply RANSAC to choose best H
        # choose random 4 pairs of matching points
        N = 4
        best_H = None
        best_inliers = 0
        for _ in range(1500):
            idx = random.sample(range(len(matches)), N)
            v = np.array([kp1[m.queryIdx].pt for m in [matches[i] for i in idx]])
            u = np.array([kp2[m.trainIdx].pt for m in [matches[i] for i in idx]])
            H = solve_homography(u, v)
            inliers = 0
            for m in matches:
                v = kp1[m.queryIdx].pt
                u = kp2[m.trainIdx].pt
                v = np.array([v[0], v[1], 1])
                u = np.array([u[0], u[1], 1])
                v_ = np.dot(H, u)
                v_ = np.array([v_[0]/v_[2], v_[1]/v_[2]])
                if np.linalg.norm(v[:2] - v_) < 3:
                    inliers += 1
            if inliers > best_inliers:
                best_inliers = inliers
                best_H = H
        
        # TODO: 3. chain the homographies
        last_best_H = np.dot(last_best_H, best_H)
        # TODO: 4. apply warping
        dst = warping(im2, dst, last_best_H, 0, h_max, 0, w_max, direction='b')
    out = dst
    return out 

if __name__ == "__main__":
    # ================== Part 4: Panorama ========================
    # TODO: change the number of frames to be stitched
    FRAME_NUM = 3
    imgs = [cv2.imread('../resource/frame{:d}.jpg'.format(x)) for x in range(1, FRAME_NUM + 1)]
    output4 = panorama(imgs)
    cv2.imwrite('output4.png', output4)
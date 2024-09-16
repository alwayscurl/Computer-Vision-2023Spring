import numpy as np
import cv2.ximgproc as xip
import cv2

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both Il to Ir and Ir to Il for later left-right consistency
    
    wid = 3
    Il_pad = cv2.copyMakeBorder(Il, wid//2, wid//2, wid//2+max_disp, wid//2+max_disp, cv2.BORDER_REFLECT)
    Ir_pad = cv2.copyMakeBorder(Ir, wid//2, wid//2, wid//2+max_disp, wid//2+max_disp, cv2.BORDER_REFLECT)
    
    Il_bin = np.zeros((h, w+2*max_disp, 3, wid*wid), dtype=int)
    Ir_bin = np.zeros((h, w+2*max_disp, 3, wid*wid), dtype=int)
    
    lcosts = np.zeros((h, w, max_disp))
    rcosts = np.zeros((h, w, max_disp))
    idx = 0
    for y in range(-(wid//2), wid//2+1):
        for x in range(-(wid//2), wid//2+1):
            Il_mask = Il_pad[wid//2:wid//2+h, wid//2:wid//2+w+2*max_disp] > Il_pad[wid//2+x:wid//2+x+h, wid//2+y:wid//2+y+w+2*max_disp]
            Ir_mask = Ir_pad[wid//2:wid//2+h, wid//2:wid//2+w+2*max_disp] > Ir_pad[wid//2+x:wid//2+x+h, wid//2+y:wid//2+y+w+2*max_disp]
            Il_bin[Il_mask, idx] += 1
            Ir_bin[Ir_mask, idx] += 1
            idx += 1
            
    for disp in range(max_disp):
        lcost = np.bitwise_xor(Il_bin[:, max_disp:max_disp+w], Ir_bin[:, max_disp-disp:max_disp+w-disp])
        lcost = np.sum(lcost, axis=3)
        lcost = np.sum(lcost, axis=2)
        lcosts[:,:,disp] = lcost
        
        rcost = np.bitwise_xor(Ir_bin[:, max_disp:max_disp+w], Il_bin[:, max_disp+disp:max_disp+w+disp])
        rcost = np.sum(rcost, axis=3)
        rcost = np.sum(rcost, axis=2)
        rcosts[:,:,disp] = rcost
        
    lcosts = lcosts.astype(np.float32)
    rcosts = rcosts.astype(np.float32)
    
    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparity)
    d = 10              # Diameter of each pixel neighborhood
    sigmaColor = 20     # Filter sigma in the color space
    sigmaSpace = 15     # Filter sigma in the coordinate space
    
    for idx in range(max_disp):
        lcost = lcosts[:, :, idx]
        lcosts[:, :, idx] = xip.jointBilateralFilter(Il, lcost, d, sigmaColor, sigmaSpace).reshape((h, w))
        rcost = rcosts[:, :, idx]
        rcosts[:, :, idx] = xip.jointBilateralFilter(Ir, rcost, d, sigmaColor, sigmaSpace).reshape((h, w))
        
    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    
    llabels = np.argmin(lcosts, axis=2)
    rlabels = np.argmin(rcosts, axis=2)
    
    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    
    holes = np.zeros(llabels.shape, dtype=np.bool)
    for x in range(w):
        for y in range(h):
            if x - llabels[y, x] < 0:
                holes[y, x] = True
            elif llabels[y, x] != rlabels[y, x - llabels[y, x]]:
                holes[y, x] = True
            else:
                labels[y, x] = llabels[y, x]
    
    for x in range(w):
        for y in range(h):
            if holes[y, x]:
                l, lvalid, r, rvalid = 1, max_disp, 1, max_disp
                while x-l >= 0:
                    if not holes[y, x-l]:
                        lvalid = llabels[y, x-l]
                        break
                    l += 1
                while x+r < w:
                    if not holes[y, x+r]:
                        rvalid = llabels[y, x+r]
                        break
                    r += 1
                labels[y, x] = min(lvalid, rvalid)
    
    # Applying more weighted median filter will improve performance
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), labels, 20, sigma=15)
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), labels, 10, sigma=20)
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), labels, 3, sigma=40)
    
    return labels.astype(np.uint8)


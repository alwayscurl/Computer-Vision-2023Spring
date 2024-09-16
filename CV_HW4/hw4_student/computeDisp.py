import numpy as np
import cv2
import cv2.ximgproc as xip

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # Compute matching cost using Census Transform and Hamming distance
    kernel_size = 5
    offset = kernel_size // 2
    Il_padding = np.pad(Il, ((offset, offset), (offset+max_disp, offset+max_disp), (0, 0)), 'edge')
    Ir_padding = np.pad(Ir, ((offset, offset), (offset+max_disp, offset+max_disp), (0, 0)), 'edge')
    
    Il_census = np.zeros((h, w+2*max_disp, 3, kernel_size*kernel_size-1), dtype=int)
    Ir_census = np.zeros((h, w+2*max_disp, 3, kernel_size*kernel_size-1), dtype=int)
    
    cost_l = np.zeros((h, w, max_disp))
    cost_r = np.zeros((h, w, max_disp))
    idx = 0
    for y in range(-offset, offset+1):
        for x in range(-offset, offset+1):
            if x == 0 and y == 0:
                continue
            Il_mask = Il_padding[offset:h+offset, offset:w+offset+2*max_disp] > Il_padding[offset+x:h+offset+x, offset+y:w+offset+y+2*max_disp]
            Ir_mask = Ir_padding[offset:h+offset, offset:w+offset+2*max_disp] > Ir_padding[offset+x:h+offset+x, offset+y:w+offset+y+2*max_disp]
            Il_census[Il_mask, idx] += 1
            Ir_census[Ir_mask, idx] += 1
            idx += 1
            
    for d in range(max_disp):
        cost = np.bitwise_xor(Il_census[:, max_disp:max_disp+w], Ir_census[:, max_disp-d:max_disp+w-d])
        cost = np.sum(cost, axis=3)
        cost = np.sum(cost, axis=2)
        cost_l[:, :, d] = cost
        
        cost = np.bitwise_xor(Ir_census[:, max_disp:max_disp+w], Il_census[:, max_disp+d:max_disp+w+d])
        cost = np.sum(cost, axis=3)
        cost = np.sum(cost, axis=2)
        cost_r[:, :, d] = cost
        
    cost_l = cost_l.astype(np.float32)
    cost_r = cost_r.astype(np.float32)
        
    # >>> Cost Aggregation
    # Refine the cost using Joint Bilateral Filter
    sigma_s = 15
    sigma_r = 10
    for d in range(max_disp):
        cost_l[:, :, d] = xip.jointBilateralFilter(Il, cost_l[:, :, d], 10, sigma_s, sigma_r)
    for d in range(max_disp):
        cost_r[:, :, d] = xip.jointBilateralFilter(Ir, cost_r[:, :, d], 10, sigma_s, sigma_r)
        
    # >>> Disparity Optimization
    # Determine disparity based on the minimum cost (Winner-Take-All)
    
    labels_l = np.argmin(cost_l, axis=2)
    labels_r = np.argmin(cost_r, axis=2)
    
    # >>> Disparity Refinement
    # Enhance the disparity map using left-right consistency check, hole filling, and weighted median filtering
    
    # Optional: Fill holes and apply weighted median filtering for better refinement
    # This can be customized further as per specific requirements or additional techniques

    hole_detect = np.zeros(labels_l.shape, dtype=np.bool)
    for x in range(w):
        for y in range(h):
            if x - labels_l[y, x] < 0:
                hole_detect[y, x] = True
            elif labels_l[y, x] != labels_r[y, x - labels_l[y, x]]:
                hole_detect[y, x] = True
            else:
                labels[y, x] = labels_l[y, x]
                
    for x in range(w):
        for y in range(h):
            if hole_detect[y, x]:
                l_disp, r_disp = 1, 1
                lvalid, rvalid = max_disp, max_disp
                while x-l_disp >= 0:
                    if hole_detect[y, x-l_disp] == False:
                        lvalid = labels_l[y, x-l_disp]
                        break
                    l_disp += 1
                while x+r_disp < w:
                    if hole_detect[y, x+r_disp] == False:
                        rvalid = labels_l[y, x+r_disp]
                        break
                    r_disp += 1
                labels[y, x] = min(lvalid, rvalid)
    
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), labels, 15, sigma=15)
    
    return labels.astype(np.uint8)
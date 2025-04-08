import numpy as np
from numpy import matlib
import os
import math
import random

import fabio
import warnings
import logging

# library for patch extraction
import cv2, os, h5py

import numpy as np
import cv2

def frame_peak_patches_cv2_optimized(frame, psz, angle, min_intensity=0, max_r=None):
    """
    An optimized version of frame_peak_patches_cv2 that:
      - Vectorizes dimension checks (width, height) and radius checks
      - Minimizes repeated slicing and padding calls
      - Skips small or large or out-of-range components early
    """
    fh, fw = frame.shape
    
    # Build a simple binary mask for connected components
    mask = (frame > min_intensity).astype(np.uint8)
    comps, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    # stats columns: [LEFT, TOP, WIDTH, HEIGHT, AREA]
    # centroids:     (x, y)

    # Extract the widths & heights from the stats (ignore comp = 0)
    widths  = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    
    # 1) Skip single‐pixel (or too small) components
    # 2) Skip components bigger than patch size
    too_small_mask = (widths < 3) | (heights < 3)
    too_large_mask = (widths > psz) | (heights > psz)
    valid_dims_mask = ~(too_small_mask | too_large_mask)

    # Optional: also skip if centroid is outside max radius
    if max_r is not None:
        # Distance from frame center
        xs = centroids[1:, 0] - (fw / 2)
        ys = centroids[1:, 1] - (fh / 2)
        dist2 = xs*xs + ys*ys
        in_radius_mask = dist2 <= (max_r**2)
        valid_dims_mask &= in_radius_mask

    # Figure out which component indices survive all checks
    valid_comp_indices = np.where(valid_dims_mask)[0] + 1  # offset by 1

    # For reporting how many big / small peaks we skipped
    big_peaks = np.count_nonzero(too_large_mask)
    small_pixel_peaks = np.count_nonzero(too_small_mask)

    patches = []
    peak_ori = []

    # Iterate only over the surviving components
    for comp in valid_comp_indices:
        # Bounding box
        col_s = stats[comp, cv2.CC_STAT_LEFT]
        col_e = col_s + stats[comp, cv2.CC_STAT_WIDTH]
        row_s = stats[comp, cv2.CC_STAT_TOP]
        row_e = row_s + stats[comp, cv2.CC_STAT_HEIGHT]

        # Slice out the region and zero out everything but "comp"
        local_cc = cc_labels[row_s:row_e, col_s:col_e]
        _patch = frame[row_s:row_e, col_s:col_e]
        _mask  = (local_cc == comp)
        _patch = _patch * _mask  # zero out non‐comp pixels

        # Pad if needed (only if patch is smaller than psz × psz)
        h, w = _patch.shape
        if h != psz or w != psz:
            # Compute how much to pad on each side
            _lp = (psz - w) // 2
            _rp = (psz - w) - _lp
            _tp = (psz - h) // 2
            _bp = (psz - h) - _tp
            _patch = np.pad(
                _patch,
                pad_width=((_tp, _bp), (_lp, _rp)),
                mode='constant',
                constant_values=0
            )
        else:
            # No padding was done
            _tp, _lp = 0, 0

        # Check if patch is uniform (all zero or all same value)
        # If min == max, it’s not interesting
        if _patch.min() == _patch.max():
            continue

        # Compute the offset of the patch in the original image
        _pr_o = row_s - _tp
        _pc_o = col_s - _lp
        peak_ori.append((angle, _pr_o, _pc_o))
        patches.append(_patch)

    # Convert to an array once at the end
    patches = np.array(patches, dtype=np.float16)
    peak_ori = np.array(peak_ori, dtype=np.float32)  # or float64, depending on your preference

    return patches, peak_ori, big_peaks



def frame_peak_patches_cv2(frame, psz, angle, min_intensity=0, max_r=None):
    fh, fw = frame.shape
    patches, peak_ori = [], []
    mask = (frame > min_intensity).astype(np.uint8)
    comps, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    
    big_peaks = 0
    small_pixel_peak = 0
    for comp in range(1, comps):
        # ignore single-pixel peak
        if stats[comp, cv2.CC_STAT_WIDTH] < 3 or stats[comp, cv2.CC_STAT_HEIGHT] < 3: 
            small_pixel_peak += 1
            continue 
            
        # ignore component that is bigger than patch size
        if stats[comp, cv2.CC_STAT_WIDTH] > psz or stats[comp, cv2.CC_STAT_HEIGHT] > psz:
            big_peaks += 1
            continue
        
        # check if the component is within the max radius
        c, r = centroids[comp, 0], centroids[comp, 1]
        if max_r is not None and max_r**2 < ((c - fw/2)**2 + ( r - fh/2)**2):
            continue
                    
        col_s = stats[comp, cv2.CC_STAT_LEFT]
        col_e = col_s + stats[comp, cv2.CC_STAT_WIDTH]
        
        row_s = stats[comp, cv2.CC_STAT_TOP]
        row_e = row_s + stats[comp, cv2.CC_STAT_HEIGHT]

        _patch = frame[row_s:row_e, col_s:col_e]
        
        # mask out other labels in the patch
        _mask  = cc_labels[row_s:row_e, col_s:col_e] == comp
        _patch = _patch * _mask

        if _patch.size != psz * psz:
            h, w = _patch.shape
            _lp = (psz - w) // 2
            _rp = (psz - w) - _lp
            _tp = (psz - h) // 2
            _bp = (psz - h) - _tp
            _patch = np.pad(_patch, ((_tp, _bp), (_lp, _rp)), mode='constant', constant_values=0)
        else:
            _tp, _lp = 0, 0

        _min, _max = _patch.min(), _patch.max()
        if _min == _max: continue

        _pr_o = row_s - _tp
        _pc_o = col_s - _lp
        peak_ori.append((angle, _pr_o, _pc_o))
        patches.append(_patch)

    return np.array(patches).astype(np.float16), np.array(peak_ori), big_peaks

def ge_raw2array(gefname, skip_frm=0):
    det_res = 2048
    frame_sz= det_res * det_res * 2 # the 2 is the size of uint16
    head_sz = 8192 + skip_frm * frame_sz # skip frames as needed
    n_frame = int((os.stat(gefname).st_size - head_sz) / frame_sz)
    mod = (os.stat(gefname).st_size - head_sz) % frame_sz
    if mod != 0:
        print("data in the file are not completely parsed, %d left over" % mod)
    
    print(f"n_frame: {n_frame}, head_sz: {head_sz}")
    
    with open(gefname, "rb") as fp:
        fp.seek(head_sz, os.SEEK_SET)
        # frames = np.zeros((n_frame, det_res, det_res), dtype=np.uint16)
        frames = np.fromfile(fp, dtype=np.uint16, count=n_frame*det_res*det_res).reshape((n_frame, det_res, det_res))

    print(f"NONfabio frames dimension: {frames.shape}")

    return frames

def ge_raw2array_fabio(gefname, skip_frm=0):

    # add this line to suppress some warnings
    logging.getLogger("fabio").setLevel(logging.ERROR)

    # Load the image file
    image = fabio.open(gefname)

    # Check if the file supports multiple frames
    try:
        nframes = int(image.nframes)  # AttributeError if not supported
        print("Number of frames:", nframes)
    except AttributeError:
        nframes = 1
        print("Number of frames:", nframes)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)  # Replace with the relevant category
        frames = [image.get_frame(i).data for i in range(skip_frm, nframes)]

    # Convert the list of frames to a 3D NumPy array
    frames_array = np.array(frames)

    print("fabio frames_array_dimension: ", frames_array.shape)
    # print("frames_array: ", frames_array)

    return frames_array

def ge_raw2patch(gefname, ofn, dark, thold, psz, skip_frm=0, min_intensity=0, max_r=None):
    frames = ge_raw2array(gefname, skip_frm=1)

    print(f"ge_raw2_path frames dimension: {frames.shape}")

    if not isinstance(dark, str):
        print(f"dark file dimension: {dark.shape}")
        frames = frames.astype(np.float32) - dark

    if thold > 0:
        frames[frames < thold] = 0
    frames = frames.astype(np.uint16)

    print("finish subtraction of the dark")
    
    patches, peak_ori = [], []
    frames_idx = []

    too_big_peaks = 0
    for i in range(frames.shape[0]):
        _pc, _ori, _bp = frame_peak_patches_cv2_optimized(frames[i], angle=i, psz=psz, min_intensity=0, max_r=None)
        if(_pc.size == 0):
            continue
        patches.append(_pc)
        peak_ori.append(_ori)
        frames_idx.append([i] * _pc.shape[0])
        too_big_peaks += _bp

    patches = np.concatenate(patches,  axis=0)
    peak_ori= np.concatenate(peak_ori, axis=0)
    frames_idx = np.concatenate(frames_idx, axis=0)

    print(f"{patches.shape[0]} patches of size {psz} cropped from {gefname}, {too_big_peaks} are too big.")
    with h5py.File(ofn, 'w') as h5fd:
        h5fd.create_dataset('patch', data=patches, dtype=np.uint16)
        h5fd.create_dataset('coordinate', data=peak_ori, dtype=np.uint16)
        h5fd.create_dataset('frame_idx', data=frames_idx, dtype=np.uint16)

# scan from wdir and return a list of datasets
def find_dataset_pooling(dataDir, thold, datasetPre):
    
    listFiles = os.listdir(dataDir)
    
    filesString   = []
    filesPressure = []
    filesIdx      = []

    numDatasets = 0 
    for fileString in listFiles:
        if fileString.startswith(datasetPre):
            print(fileString)
            filesString.append(fileString)
            #pressure = int(fileString[14:16])
            #filesPressure.append(pressure)
            x = fileString.split("_")
            idx = int(x[-1].split(".")[0])
            #idx = int(fileString[23:29])
            filesIdx.append(idx)
            numDatasets += 1

    print(f"There are {numDatasets} patch datasets in total")
    #print(filesString)
    #print(filesPressure)
    #print(filesIdx) 

    return filesString, filesPressure, filesIdx


# finish a function here to 
def find_dataset_single(idata, idark, thold, datasetPre, outputFile=None):

    print(f"Raw file based mode is enabled, now need to process {idata} and substract it by the dark file {idark} if provided")

    listFiles = [str(idata)]

    # need to add the function to subtract the dark file, patch extraction and output to a h5 file
    # read the raw scan and dark file and output a h5 file for later processing    
    if idark != "dark":
        print(f"Reading dark file from {idark} ... ")
        dark = ge_raw2array_fabio(idark, skip_frm=0).mean(axis=0).astype(np.float32)
        print(f"Done with reading dark file from {idark}")
    else:
        print(f"no dark file provided, skip dark file reading")


    outFile = f"{datasetPre}.h5" if outputFile is None else outputFile
    print(f'the output h5 file is:{outFile}')

    print(f"Reading baseline/testing file from {idata} ... ")    
    if idark != "dark":
        ge_raw2patch(gefname=idata, ofn=outFile, dark=dark, thold=thold, psz=15, skip_frm=0, \
                    min_intensity=0, max_r=None)
    else:
        ge_raw2patch(gefname=idata, ofn=outFile, dark=idark, thold=thold, psz=15, skip_frm=0, \
                         min_intensity=0, max_r=None)
    print(f"Done with reading baseline/training file from {idata}")

    filesString   = []
    filesPressure = []
    filesIdx      = []

    numDatasets = 0 
    for fileString in listFiles:
        if fileString.startswith(datasetPre):
            print(fileString)
            filesString.append(fileString)
            #pressure = int(fileString[14:16])
            #filesPressure.append(pressure)
            x = fileString.split("_")
            idx = int(x[-1].split(".")[0])
            #idx = int(fileString[23:29])
            filesIdx.append(idx)
            numDatasets += 1

    print(f"There are {numDatasets} patch datasets in total")
    #print(filesString)
    #print(filesPressure)
    #print(filesIdx) 
    # outFile = '/home/beams/WZHENG/RareEventDetectionHEDM/code/sam9_all_init.edf.h5'

    return [str(outFile)], filesPressure, filesIdx


# find a list of patches from a dataset based on the degree range
# inputs: frame_idx: a list of frame index
#         degree: the degree range
#         seed: random seed to control the starting degree
# outputs: a list of patch index

def find_degree_pathches(num_patches, frame_idx, degree, seed=0, degs_mode=1):

    # first is to find the frame index range
    frameKeys = []
    frameDict = {}
    patchIdx = 0
    for frameIdx in frame_idx:
        frameIdx = int(frameIdx)
        if frameIdx not in frameKeys:
            frameKeys.append(frameIdx)
            frameDict[frameIdx] = []
            frameDict[frameIdx].append(patchIdx)
        else:
            frameDict[frameIdx].append(patchIdx)
        patchIdx += 1
    
    # find the average number of frames per degree
    numFrames = max(frameKeys) + 1
    avenumFramesperDegree = math.ceil(numFrames/360)
    avenumPatchesperDegree = math.ceil(num_patches/360)

    random.seed(seed)
    startDeg = random.randrange(0, 360)

    selectDeg = startDeg
    patchIdxs = []

    startFrame = selectDeg * avenumFramesperDegree
    while startFrame not in frameKeys:
        startFrame += 1
        # the following code is used to avoid overflow for starting degree
        if startFrame > max(frameKeys):
            startFrame = 0
    frameIdx = frameKeys.index(startFrame)
    frame = frameKeys[frameIdx]

    frameIdxSteps = 0
    frameSteps = 0

    # use a flag here to indicate whether we encounterd some dark field
    # if deg_mode == 0, we will discard those sampling results with dark field
    darkOrNot = 0
    count     = 0
    while len(patchIdxs) <= avenumPatchesperDegree*degree:
        # extract all patches in this frame
        patchIdxs += frameDict[frame]
        
        frameIdx += 1
        frameIdxSteps += 1

        # dark field should not happend at start
        if frameIdx >= len(frameKeys): 
            frameIdx -= len(frameKeys)
            frame = frameKeys[frameIdx]
            frameIdxSteps = 0 
            frameSteps    = 0
        elif count == 0:
            frame = frameKeys[frameIdx]              
            frameSteps += 1
        else: 
            oldFrame = frame
            frame = frameKeys[frameIdx]              
            frameSteps += frame
            frameSteps -= oldFrame 

        count += 1

        if frameSteps != frameIdxSteps:
            darkOrNot = 1
            # print(f"a dark file is found at frame {frame} with start {startFrame} and {avenumFramesperDegree*5}")

    return patchIdxs, darkOrNot


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def str2tuple(s):
    return tuple(s.split('_'))

def s2ituple(s):
    return tuple(int(_s) for _s in s.split('_'))
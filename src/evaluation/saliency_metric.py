from skimage.transform import resize
import numpy as np
from skimage import img_as_float
from skimage import exposure
#import cv2 as cv
from evaluation.multimatch import *

def saliency_map_metric(logits, tgt_pos):
	# logits: seq_len, package_size
	# tgt_pos: seq_len
	logits = logits.detach().cpu().numpy()
	tgt_pos = tgt_pos.detach().cpu().numpy()
	seq_len, package_size = logits.shape[0], logits.shape[1]
	sim_total = 0
	for i in range(seq_len):
		saliency_map = logits[i]
		fixation_map = np.zeros(package_size)
		fixation_map[tgt_pos[i]] = 1
		sim = SIM(saliency_map, fixation_map)
		sim_total += sim
	return sim_total/seq_len

def normalize(x, method='standard', axis=None):
    '''Normalizes the input with specified method.
    Parameters
    ----------
    x : array-like
    method : string, optional
        Valid values for method are:
        - 'standard': mean=0, std=1
        - 'range': min=0, max=1
        - 'sum': sum=1
    axis : int, optional
        Axis perpendicular to which array is sliced and normalized.
        If None, array is flattened and normalized.
    Returns
    -------
    res : numpy.ndarray
        Normalized array.
    '''
    # TODO: Prevent divided by zero if the map is flat
    x = np.array(x, copy=False)
    if axis is not None:
        y = np.rollaxis(x, axis).reshape([x.shape[axis], -1])
        shape = np.ones(len(x.shape))
        shape[axis] = x.shape[axis]
        if method == 'standard':
            res = (x - np.mean(y, axis=1).reshape(shape)) / np.std(y, axis=1).reshape(shape)
        elif method == 'range':
            res = (x - np.min(y, axis=1).reshape(shape)) / (np.max(y, axis=1) - np.min(y, axis=1)).reshape(shape)
        elif method == 'sum':
            res = x / np.float_(np.sum(y, axis=1).reshape(shape))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    else:
        if method == 'standard':
            res = (x - np.mean(x)) / np.std(x)
        elif method == 'range':
            res = (x - np.min(x)) / (np.max(x) - np.min(x))
        elif method == 'sum':
            res = x / float(np.sum(x))
        else:
            raise ValueError('method not in {"standard", "range", "sum"}')
    return res


def match_hist(image, cdf, bin_centers, nbins=256):
    '''Modify pixels of input image so that its histogram matches target image histogram, specified by:
    cdf, bin_centers = cumulative_distribution(target_image)
    Parameters
    ----------
    image : array
        Image to be transformed.
    cdf : 1D array
        Values of cumulative distribution function of the target histogram.
    bin_centers ; 1D array
        Centers of bins of the target histogram.
    nbins : int, optional
        Number of bins for image histogram.
    Returns
    -------
    out : float array
        Image array after histogram matching.
    References
    ----------
    [1] Matlab implementation histoMatch(MTX, N, X) by Simoncelli, 7/96.
    '''
    image = img_as_float(image)
    old_cdf, old_bin = exposure.cumulative_distribution(image, nbins) # Unlike [1], we didn't add small positive number to the histogram
    new_bin = np.interp(old_cdf, cdf, bin_centers)
    out = np.interp(image.ravel(), old_bin, new_bin)
    return out.reshape(image.shape)

def NSS(saliency_map, fixation_map):
	'''
	Normalized scanpath saliency of a saliency map,
	defined as the mean value of normalized (i.e., standardized) saliency map at fixation locations.
	You can think of it as a z-score. (Larger value implies better performance.)
	Parameters
	----------
	saliency_map : real-valued matrix
		If the two maps are different in shape, saliency_map will be resized to match fixation_map..
	fixation_map : binary matrix
		Human fixation map (1 for fixated location, 0 for elsewhere).
	Returns
	-------
	NSS : float, positive
	'''
	s_map = np.array(saliency_map, copy=False)
	f_map = np.array(fixation_map, copy=False) > 0.5
	if s_map.shape != f_map.shape:
		s_map = resize(s_map, f_map.shape)
	# Normalize saliency map to have zero mean and unit std
	s_map = normalize(s_map, method='standard')
	# Mean saliency value at fixation locations
	return np.mean(s_map[f_map])

def CC(saliency_map1, saliency_map2):
	'''
	Pearson's correlation coefficient between two different saliency maps
	(CC=0 for uncorrelated maps, CC=1 for perfect linear correlation).
	Parameters
	----------
	saliency_map1 : real-valued matrix
		If the two maps are different in shape, saliency_map1 will be resized to match saliency_map2.
	saliency_map2 : real-valued matrix
	Returns
	-------
	CC : float, between [-1,1]
	'''
	map1 = np.array(saliency_map1, copy=False)
	map2 = np.array(saliency_map2, copy=False)
	if map1.shape != map2.shape:
		map1 = resize(map1, map2.shape, order=3, mode='nearest') # bi-cubic/nearest is what Matlab imresize() does by default
	# Normalize the two maps to have zero mean and unit std
	map1 = normalize(map1, method='standard')
	map2 = normalize(map2, method='standard')
	# Compute correlation coefficient
	return np.corrcoef(map1.ravel(), map2.ravel())[0,1]

def SIM(saliency_map1, saliency_map2):
	'''
	Similarity between two different saliency maps when viewed as distributions
	(SIM=1 means the distributions are identical).
	This similarity measure is also called **histogram intersection**.
	Parameters
	----------
	saliency_map1 : real-valued matrix
		If the two maps are different in shape, saliency_map1 will be resized to match saliency_map2.
	saliency_map2 : real-valued matrix
	Returns
	-------
	SIM : float, between [0,1]
	'''
	map1 = np.array(saliency_map1, copy=False)
	map2 = np.array(saliency_map2, copy=False)
	'''if map1.shape != map2.shape:
		map1 = resize(map1, map2.shape, order=3, mode='nearest') # bi-cubic/nearest is what Matlab imresize() does by default
	# Normalize the two maps to have values between [0,1] and sum up to 1
	map1 = normalize(map1, method='range')
	map2 = normalize(map2, method='range')
	map1 = normalize(map1, method='sum')
	map2 = normalize(map2, method='sum')'''
	# Compute histogram intersection
	intersection = np.minimum(map1, map2)
	return np.sum(intersection)

'''def EMD(saliency_map1, saliency_map2, sub_sample=1/32.0):
	
	Earth Mover's Distance measures the distance between two probability distributions
	by how much transformation one distribution would need to undergo to match another
	(EMD=0 for identical distributions).
	Parameters
	----------
	saliency_map1 : real-valued matrix
		If the two maps are different in shape, saliency_map1 will be resized to match saliency_map2.
	saliency_map2 : real-valued matrix
	Returns
	-------
	EMD : float, positive
	map2 = np.array(saliency_map2, copy=False)
	# Reduce image size for efficiency of calculation
	map2 = resize(map2, np.round(np.array(map2.shape)*sub_sample), order=3, mode='nearest')
	map1 = resize(saliency_map1, map2.shape, order=3, mode='nearest')
	# Histogram match the images so they have the same mass
	map1 = match_hist(map1, *exposure.cumulative_distribution(map2))
	# Normalize the two maps to sum up to 1,
	# so that the score is independent of the starting amount of mass / spread of fixations of the fixation map
	map1 = normalize(map1, method='sum')
	map2 = normalize(map2, method='sum')
	# Compute EMD with OpenCV
	# - http://docs.opencv.org/modules/imgproc/doc/histograms.html#emd
	# - http://stackoverflow.com/questions/5101004/python-code-for-earth-movers-distance
	# - http://stackoverflow.com/questions/12535715/set-type-for-fromarray-in-opencv-for-python
	r, c = map2.shape
	x, y = np.meshgrid(range(c), range(r))
	signature1 = cv.CreateMat(r*c, 3, cv.CV_32FC1)
	signature2 = cv.CreateMat(r*c, 3, cv.CV_32FC1)
	cv.Convert(cv.fromarray(np.c_[map1.ravel(), x.ravel(), y.ravel()]), signature1)
	cv.Convert(cv.fromarray(np.c_[map2.ravel(), x.ravel(), y.ravel()]), signature2)
	return cv.CalcEMD2(signature2, signature1, cv.CV_DIST_L2)'''

def normalize_map(s_map):
	# normalize the salience map (as done in MIT code)
	norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
	return norm_s_map

def discretize_gt(gt):
	import warnings
	warnings.warn('can improve the way GT is discretized')
	return gt/255

def infogain(s_map,gt,baseline_map):
	gt = discretize_gt(gt)
	# assuming s_map and baseline_map are normalized
	eps = 2.2204e-16

	s_map = s_map/(np.sum(s_map)*1.0)
	baseline_map = baseline_map/(np.sum(baseline_map)*1.0)

	# for all places where gt=1, calculate info gain
	temp = []
	x,y = np.where(gt==1)
	for i in zip(x,y):
		temp.append(np.log2(eps + s_map[i[0],i[1]]) - np.log2(eps + baseline_map[i[0],i[1]]))

	return np.mean(temp)


def zero_one_similarity(a, b):
    if a == b:
        return 1.0
    else:
        return 0.0


def nw_matching(pred_string, gt_string, gap=0.0):
    # NW string matching with zero_one_similarity
    F = np.zeros((len(pred_string) + 1, len(gt_string) + 1), dtype=np.float32)
    for i in range(1 + len(pred_string)):
        F[i, 0] = gap * i
    for j in range(1 + len(gt_string)):
        F[0, j] = gap * j
    for i in range(1, 1 + len(pred_string)):
        for j in range(1, 1 + len(gt_string)):
            a = pred_string[i - 1]
            b = gt_string[j - 1]
            match = F[i - 1, j - 1] + zero_one_similarity(a, b)
            delete = F[i - 1, j] + gap
            insert = F[i, j - 1] + gap
            F[i, j] = np.max([match, delete, insert])
    score = F[len(pred_string), len(gt_string)]
    return score / max(len(pred_string), len(gt_string))

def compare_multi_gazes(gt, gaze):
	# gt: gt_len, 1; gaze: num, pred_len
	gt = gt.detach().cpu().numpy()[:, 0]
	gaze = gaze #.detach().cpu().numpy()
	total_ss = 0
	total_gaze = len(gaze)
	for i in range(total_gaze):
		gaze_ = gaze[i].detach().cpu().numpy()
		ss = nw_matching(gaze_, gt)
		total_ss += ss
	return total_ss / total_gaze

def compare_mm(gt, gaze,col_num, row_num):
	# gt: gt_len, 1; gaze: num, pred_len
	gt = gt.detach().cpu().numpy()[:, 0]
	gaze = gaze #.detach().cpu().numpy()
	total_mm = 0
	total_gaze = len(gaze)
	for i in range(total_gaze):
		gaze_ = gaze[i].detach().cpu().numpy()
		mm = docomparison(gaze_,gt,col_num,row_num)
		total_mm += np.mean(mm)
	return total_mm / total_gaze
import os
import os.path as osp
import numpy as np
from time import strftime, localtime
from easydict import EasyDict as edict

__C = edict()
cfg = __C

__C.ANCHOR_SCALES = [16]

# Default GPU device id
__C.GPU_ID = 0

#平均像素值
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])



__C.TEST = edict()
__C.TEST.checkpoints_path = "/home/phoebushe/OCR/text-detection-ctpn-master/output/ctpn_end2end/voc_2007_trainval/"
__C.TEST.DETECT_MODE = "H"#H/O for horizontal/oriented mode
# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Propose boxes
__C.TEST.HAS_RPN = True

# Test using these proposals
__C.TEST.PROPOSAL_METHOD = 'selective_search'

## NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
## Number of top scoring boxes to keep before apply NMS to RPN proposals
#__C.TEST.RPN_PRE_NMS_TOP_N = 6000
__C.TEST.RPN_PRE_NMS_TOP_N = 12000
## Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 1000
#__C.TEST.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 8
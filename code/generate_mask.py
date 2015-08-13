# Compress a caffemodel

import numpy as np
import argparse
import sys
import time

caffe_root = '/home/master/02/joel1211/caffe/'
sys.path.insert(1, caffe_root + 'python')

import caffe

# Parse args
parser = argparse.ArgumentParser(description='Generate mask for pretrained model')
parser.add_argument('-th',type=float)
parser.add_argument('-name',type=str)
args = vars(parser.parse_args())

# Path
loadModel = '/tmp3/joel1211/fc2conv/model/caffenet/bvlc_reference_caffenet.caffemodel'
prototxt = '/tmp3/joel1211/deepComp/model/deploy.prototxt'

# Load model
net = caffe.Net(prototxt, loadModel, caffe.TEST)
print 'Load from {}'.format(loadModel)

# Collect weights
fc_name = args['name']
th = args['th']

tmp_params = {fc_name: (net.params[fc_name][0].data, net.params[fc_name][1].data)}
fc_weight = net.params[fc_name][0].data
fc_weight[np.absolute(fc_weight) < th] = 0
#fc_bias = net.params[fc_name][1].data
#fc_bias[np.absolute(fc_bias) < th] = 0
mask_weight = net.params[fc_name][0].data
mask_weight[np.absolute(mask_weight) < th] = 0
mask_weight[np.absolute(mask_weight) >= th] = 1
print mask_weight
#print net.params[fc][0].data
#print net.params[fc][1].data

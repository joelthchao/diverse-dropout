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
parser.add_argument('-ratio',type=float) # ratio of zero out
parser.add_argument('-name',type=str) # name of layer
parser.add_argument('-out',type=str) # output mask file
args = vars(parser.parse_args())

# Path
loadModel = '/tmp3/joel1211/static_dropout/model/caffenet/bvlc_reference_caffenet.caffemodel'
prototxt = '/tmp3/joel1211/static_dropout/model/caffenet/deploy.prototxt'

# Load model
net = caffe.Net(prototxt, loadModel, caffe.TEST)
print 'Load from {}'.format(loadModel)

# Collect weights
fc_name = args['name']
ratio = args['ratio']
out_file = args['out']

#tmp_params = {fc_name: (net.params[fc_name][0].data, net.params[fc_name][1].data)}
#fc_weight = net.params[fc_name][0].data
#fc_weight[np.absolute(fc_weight) < th] = 0
#fc_bias = net.params[fc_name][1].data
#fc_bias[np.absolute(fc_bias) < th] = 0
mask_weight = net.params[fc_name][0].data
sort_weight = np.sort(np.absolute(mask_weight.flatten()));
th = sort_weight[int(ratio*sort_weight.size)]
print th

mask_weight[np.absolute(mask_weight) < th] = 0
mask_weight[np.absolute(mask_weight) >= th] = 1

with open(out_file, 'w') as f:
    f.write('{0} {1} '.format(mask_weight.shape[1], mask_weight.shape[0]))
    for n in mask_weight.flatten():
        f.write('{0:.0f} '.format(n))

print 'Write mask to', out_file, 'with', np.count_nonzero(mask_weight), 'nonzero'


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
parser.add_argument('-model',type=str) # model to reprune
parser.add_argument('-proto',type=str) # prototxt for reprune model
parser.add_argument('-mask',type=str) # mask for reprune model
args = vars(parser.parse_args())

# Path
#loadModel = '/tmp3/joel1211/static_dropout/model/caffenet/bvlc_reference_caffenet.caffemodel'
#prototxt = '/tmp3/joel1211/static_dropout/model/caffenet/deploy.prototxt'
loadModel = args['model'] #'/home/master/02/joel1211/git_project/diverse-dropout/chao_test_sdfc_5_5_5_iter_40000.caffemodel'
prototxt = args['proto'] #'/home/master/02/joel1211/git_project/diverse-dropout/zoo/chao_test_sdfc_5_5_5/train_val.prototxt'

# Load model
net = caffe.Net(prototxt, loadModel, caffe.TEST)
print 'Load from {}'.format(loadModel)

# Collect weights
fc_name = args['name']
ratio = args['ratio']
out_file = args['out']
origin_mask = args['mask']

with open(origin_mask) as f:
    array = []
    for line in f: # read rest of lines
        array = [int(x) for x in line.split()]
del array[0] # Remove network input size
del array[0] # Remove network output size
mask_array = np.array(array)

#tmp_params = {fc_name: (net.params[fc_name][0].data, net.params[fc_name][1].data)}
#fc_weight = net.params[fc_name][0].data
#fc_weight[np.absolute(fc_weight) < th] = 0
#fc_bias = net.params[fc_name][1].data
#fc_bias[np.absolute(fc_bias) < th] = 0
mask_weight = net.params[fc_name][0].data
w, h = mask_weight.shape
mask_weight = np.multiply(mask_weight.flatten(), mask_array)
sort_weight = np.sort(np.absolute(mask_weight));
th = sort_weight[int(ratio*sort_weight.size)]
print th

mask_weight[np.absolute(mask_weight) <= th] = 0
mask_weight[np.absolute(mask_weight) > th] = 1


with open(out_file, 'w') as f:
    f.write('{0} {1} '.format(w, h))
    for n in mask_weight.flatten():
        f.write('{0:.0f} '.format(n))

print 'Write mask to', out_file, 'with', np.count_nonzero(mask_weight), 'nonzero'


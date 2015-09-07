~/caffe/build/tools/caffe train \
-weights=/tmp3/joel1211/fc2conv/model/caffenet/bvlc_reference_caffenet.caffemodel \
-solver=/home/master/02/joel1211/git_project/diverse-dropout/zoo/chao_test_sdfc3/solver.prototxt \
-gpu=1 \
2>&1 | tee /home/master/02/joel1211/git_project/diverse-dropout/log/chao_test_sdfc3.log

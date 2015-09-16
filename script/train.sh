~/caffe/build/tools/caffe train \
-weights=/tmp3/joel1211/static_dropout/model/caffenet/bvlc_reference_caffenet.caffemodel \
-solver=/home/master/02/joel1211/git_project/diverse-dropout/zoo/chao_test_sdfc6_95/solver.prototxt \
-gpu=2 \
2>&1 | tee /home/master/02/joel1211/git_project/diverse-dropout/log/chao_test_sdfc6_95.log

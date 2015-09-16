~/caffe/build/tools/caffe train \
-weights=/home/master/02/joel1211/git_project/diverse-dropout/caffe_90000.caffemodel \
-solver=/home/master/02/joel1211/git_project/diverse-dropout/zoo/chao_test_sdfc_5_5_5/solver.prototxt \
-gpu=1 \
2>&1 | tee /home/master/02/joel1211/git_project/diverse-dropout/log/chao_test_sdfc_5_5_5.log

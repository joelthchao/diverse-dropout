~/caffe/build/tools/caffe train \
-weights=/home/master/02/joel1211/git_project/diverse-dropout/chaonet_5_5_5_50000.caffemodel \
-solver=/home/master/02/joel1211/git_project/diverse-dropout/zoo/chao_reprune_5_5_5_8_8_8/solver.prototxt \
-gpu=3 \
2>&1 | tee /home/master/02/joel1211/git_project/diverse-dropout/log/chao_reprune_5_5_5_8_8_8.log

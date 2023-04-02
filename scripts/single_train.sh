OMP_NUM_THREADS=1 nohup python scripts/train.py --use_color --relation_prediction --gpu 1 --epoch 80 --prepare_epoch 10 > train.log 2>&1 &

# detection
# OMP_NUM_THREADS=1 nohup python -u -m torch.distributed.launch --nproc_per_node=8 scripts/train.py --use_color --relation_prediction --color_prediction --size_prediction --shape_prediction --no_reference --batch_size 12 --val_step 1 --lr 8e-3 --epoch 60 > RGB_attr+relation.log 2>&1 &

# visual grounding
# OMP_NUM_THREADS=1 nohup python -u -m torch.distributed.launch --nproc_per_node=8 scripts/train.py --use_color --relation_prediction --color_prediction --size_prediction --shape_prediction --batch_size 12 --val_step 1 --lr 8e-3 --epoch 60 > RGB_attr+relation_refer.log 2>&1 &

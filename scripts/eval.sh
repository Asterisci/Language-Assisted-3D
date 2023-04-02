# RGB detection
# OMP_NUM_THREADS=1 nohup python scripts/eval.py --folder RGB_attr+relation --detection --use_color --no_nms --force --repeat 5 --gpu 2 --batch_size 12 > eval_logs/RGB_attr+relation.log 2>&1 &

# multiview detection
# OMP_NUM_THREADS=1 nohup python scripts/eval.py --folder multiview_attr+relation --use_multiview --use_normal --detection --no_nms --force --repeat 5 --gpu 3 --batch_size 12 > eval_logs/multiview_attr+relation.log 2>&1 &

# RGB refer
# OMP_NUM_THREADS=1 nohup python scripts/eval.py --folder RGB_attr+relation_refer --reference --use_color --no_nms --force --repeat 5 --gpu 1 --batch_size 12 > eval_logs/RGB_attr+relation_refer.log 2>&1 &

# multiview refer
# OMP_NUM_THREADS=1 nohup python scripts/eval.py --folder multiview_attr+relation_refer --use_multiview --use_normal --reference --no_nms --force --repeat 5 --gpu 2 --batch_size 12 > eval_logs/multiview_attr+relation_refer.log 2>&1 &
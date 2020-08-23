CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --batchSize=4 --testBatchSize=8 --crop_height=256 --crop_width=640 --thread=4 --resume='' --shift=3 2>&1 |tee log_train.txt
#CUDA_VISIBLE_DEVICES=2 python finetune2.py --batchSize=1 --testBatchSize=8 --crop_height=240 --crop_width=528 --thread=1 --resume='' --kitti=True --shift=3 2>&1 |tee log_debug.txt

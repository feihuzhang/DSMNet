export PATH=/homes/feihu/app/usr/bin:/usr/local/cuda-11/bin:$PATH
export LD_LIBRARY_PATH=/homes/feihu/app/lib:/homes/feihu/app/cudnn/cuda/lib64:/usr/local/cuda-11/lib64:$LD_LIBRARY_PATH

__conda_setup="$('/homes/feihu/anaconda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/homes/feihu/anaconda/etc/profile.d/conda.sh" ]; then
        . "/homes/feihu/anaconda/etc/profile.d/conda.sh"
    else
        export PATH="/homes/feihu/anaconda/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate pytorch18
#python train.py --batchSize=8 --testBatchSize=8 --crop_height=320 --crop_width=1024 --thread=16 --resume='checkpoints/synthetic_epoch_3.pth' --shift=3 --stage='synthetic' --freeze_bn=1 --gpu='0,1,2,3'
python train.py --batchSize=16 --testBatchSize=8 --crop_height=256 --crop_width=640 --thread=16 --resume='checkpoints/synthetic_epoch_4.pth' --shift=3 --stage='synthetic' --freeze_bn=1 --gpu='0,1,2,3,4,5,6,7' --lr=0.00025 --save_path='checkpoints/syn256x640'
exit
python train.py --batchSize=16 --testBatchSize=8 --crop_height=320 --crop_width=1024 --thread=16 --resume='checkpoints/synthetic_epoch_3.pth' --shift=3 --stage='synthetic' --freeze_bn=1 --gpu='0,1,2,3,4,5,6,7'
exit
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --batchSize=16 --testBatchSize=8 --crop_height=256 --crop_width=640 --thread=16 --resume='' --shift=3 --stage='things' --freeze_bn=0 #2>&1 |tee log_train.txt
exit

#CUDA_VISIBLE_DEVICES=2 python finetune2.py --batchSize=1 --testBatchSize=8 --crop_height=240 --crop_width=528 --thread=1 --resume='' --kitti=True --shift=3 2>&1 |tee log_debug.txt

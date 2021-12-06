python predict.py --crop_height=384 \
                  --crop_width=1280 \
                  --max_disp=192 \
                  --data_path='/export/work/feihu/kitti2015/testing/' \
                  --test_list='lists/kitti2015_test.list' \
                  --save_path='./results/' \
                  --dataset='kitti2015' \
                  --resume='checkpoints/synthetic2_epoch_1.pth'
#                  --resume='checkpoints/sync_things.pth'
exit

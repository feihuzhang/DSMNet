python predict.py --crop_height=384 \
                  --crop_width=1280 \
                  --max_disp=192 \
                  --data_path='/media/feihu/Storage/stereo/data_scene_flow/testing/' \
                  --test_list='lists/kitti2015_test.list' \
                  --save_path='./results/' \
                  --kitti2015=1 \
                  --resume='./checkpoint/DSMNet.pth'
exit
python predict.py --crop_height=384 \
                  --crop_width=1248 \
                  --max_disp=192 \
                  --data_path='/media/feihu/Storage/stereo/data_scene_flow/testing/' \
                  --test_list='lists/kitti2015_test.list' \
                  --save_path='./norm/' \
                  --kitti2015=1 \
                  --resume='./checkpoint/DSMNet.pth'
exit
python predict.py --crop_height=1008 \
                  --crop_width=2016 \
                  --max_disp=192 \
                  --data_path='../data/cityscape/' \
                  --test_list='lists/cityscapes.list' \
                  --save_path='./results/' \
                  --kitti2015=1 \
                  --resume='./checkpoint/DSMNet.pth'
exit

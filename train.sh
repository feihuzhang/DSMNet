python train.py --batchSize=16 --testBatchSize=8 --crop_height=256 --crop_width=640 --thread=16 --resume='' --shift=3 --stage='things' --freeze_bn=1 --gpu='0,1,2,3,4,5,6,7' --lr=0.001 --save_path='checkpoints/sceneflow'
python train.py --batchSize=16 --testBatchSize=8 --crop_height=256 --crop_width=640 --thread=16 --resume='checkpoints/sceneflow.pth' --shift=3 --stage='synthetic' --freeze_bn=1 --gpu='0,1,2,3,4,5,6,7' --lr=0.001 --save_path='checkpoints/synthetic'


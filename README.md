# DSMNet

[Domain-invariant Stereo Matching Newtorks](https://arxiv.org/pdf/1911.13287.pdf)

## Oral Presentation 

[Slides](http://www.feihuzhang.com/DSMNet/DSMNet.pdf), [Video](https://youtu.be/jsLgpy5qc0s)

## Great Generalization Abilities:
DSMNet has great generalization abilities on other datasets/scenes. Models are trained only with synthetic data:

<img src="illustration/cityscapes.gif" width="720" />


## DATASET
Carla Dataset: updating ...


## Building Requirements:

    gcc: >=5.3
    GPU mem: >=5G (for testing);  >=11G (for training)
    pytorch: >=1.0
    cuda: >=9.2 (9.0 doesn’t support well for the new pytorch version and may have “pybind11 errors”.)
    tested platform/settings:
      1) ubuntu 16.04 + cuda 10.0 + python 3.6, 3.7
      2) centos + cuda 9.2 + python 3.7
      

## Install Pytorch:
You can easily install pytorch (>=1.1) by "pip install" or anaconda.


## How to Use?

Step 1: compile the libs by "sh compile.sh"
- Change the environmental variable ($PATH, $LD_LIBRARY_PATH etc.), if it's not set correctly in your system environment (e.g. .bashrc). Examples are included in "compile.sh".

Step 2: download and prepare the training dataset or your own testing set.

    download SceneFLow dataset: "FlyingThings3D", "Driving" and "Monkaa" (final pass and disparity files).
  
      -mv all training images (totallty 29 folders) into ${your dataset PATH}/frames_finalpass/TRAIN/
      -mv all corresponding disparity files (totallty 29 folders) into ${your dataset PATH}/disparity/TRAIN/
      -make sure the following 27 folders are included in the "${your dataset PATH}/disparity" "${your dataset PATH}/frames_cleanpass" and "${your dataset PATH}/frames_finalpass":
        
        15mm_focallength	35mm_focallength		TRAIN			 a_rain_of_stones_x2						
        eating_camera2_x2	eating_naked_camera2_x2		eating_x2		 family_x2			flower_storm_augmented0_x2	flower_storm_augmented1_x2
        flower_storm_x2	funnyworld_augmented0_x2	funnyworld_augmented1_x2	funnyworld_camera2_augmented0_x2	funnyworld_camera2_augmented1_x2	funnyworld_camera2_x2
        funnyworld_x2	lonetree_augmented0_x2		lonetree_augmented1_x2		lonetree_difftex2_x2		  lonetree_difftex_x2		lonetree_winter_x2
        lonetree_x2		top_view_x2			treeflight_augmented0_x2	treeflight_augmented1_x2  	treeflight_x2	
	
    download and extract Carla, kitti and kitti2015 datasets.
        
Step 3: revise parameter settings and run "train.sh" and "predict.sh" for training, finetuning and prediction/testing. Note that the “crop_width” and “crop_height” must be multiple of 64 (for "DSMNet2x2"), "max_disp" must be multiple of 16 (for "DSMNet2x2") (default: 192).


## Pretrained models:

| Sceneflow (for initialize, only 10 epochs) | Synthetic (Sceneflow + Carla) | Mixed (Real + Synthetic)|
|---|---|---|
|[Google Drive](https://drive.google.com/file/d/1oXArd2uKhZQ4SjINlHyi4OZ1RjZkkoiU/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1oIizu-ADuzKiANfpzZeNcp8FfObwAMQH/view?usp=sharing)|[Google Drive](https://drive.google.com/file/d/1ay9qD4h1lORdpk0sR3bMImvf_aS1XhnP/view?usp=sharing)|
|[Baidu Yun (password: wv6g)](https://pan.baidu.com/s/1G4ccJSAmbF0gJbf76mjObw)|[Baidu Yun (password: 7qyk)](https://pan.baidu.com/s/1bcFmwQq-ssf6dvu_XD6MJw)|[Baidu Yun (password: p6a3)](https://pan.baidu.com/s/19qF2XPI9GY7gcvjZIYy1pw)|

These pre-trained models perform a little better than those reported in the paper.
If you want to compute disparity maps on your new stereo images, "Mixed (Real + Synthetic)" would be the best choice.



## Reference:

If you find the code useful, please cite our paper:

    @inproceedings{zhang2019domaininvariant,
      title={Domain-invariant Stereo Matching Networks},
      author={Feihu Zhang and Xiaojuan Qi and Ruigang Yang and Victor Prisacariu and Benjamin Wah and Philip Torr},
      booktitle={Europe Conference on Computer Vision (ECCV)},
      year={2020}
    }

    @inproceedings{Zhang2019GANet,
      title={GA-Net: Guided Aggregation Net for End-to-end Stereo Matching},
      author={Zhang, Feihu and Prisacariu, Victor and Yang, Ruigang and Torr, Philip HS},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      pages={185--194},
      year={2019}
    }

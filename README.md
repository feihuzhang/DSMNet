# DSMNet

[Domain-invariant Stereo Matching Newtorks](https://arxiv.org/pdf/1911.13287.pdf)


<img align="center" src="http://www.feihuzhang.com/illustration/cityscape.jif">

## Oral Presentation 

[Slides](http://www.feihuzhang.com/DSMNet/DSMNet.pdf), [Video](https://youtu.be/jsLgpy5qc0s)

## DATASET
Carla Dataset:


## Building Requirements:

    gcc: >=5.3
    GPU mem: >=6.5G (for testing);  >=11G (for training, >=22G is prefered)
    pytorch: >=1.0
    cuda: >=9.2 (9.0 doesn’t support well for the new pytorch version and may have “pybind11 errors”.)
    tested platform/settings:
      1) ubuntu 16.04 + cuda 10.0 + python 3.6, 3.7
      2) centos + cuda 9.2 + python 3.7
      

## Install Pytorch:
You can easily install pytorch (>=1.0) by "pip install" to run the code. See this https://github.com/feihuzhang/GANet/issues/24

But, if you have trouble (lib conflicts) when compiling cuda libs,
installing pytorch from source would help solve most of the errors (lib conflicts).

Please refer to https://github.com/pytorch/pytorch about how to reinstall pytorch from source.

## How to Use?

Step 1: compile the libs by "sh compile.sh"
- Change the environmental variable ($PATH, $LD_LIBRARY_PATH etc.), if it's not set correctly in your system environment (e.g. .bashrc). Examples are included in "compile.sh".
- If you met the BN error, try to replace the sync-bn with another version:
    1) Install NVIDIA-Apex package https://github.com/NVIDIA/apex
          $ git clone https://github.com/NVIDIA/apex
          $ cd apex
          $ pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    2) Revise the "GANet_deep.py":
    add `import apex` 
    change all `BatchNorm2d` and `BatchNorm3d` to `apex.parallel.SyncBatchNorm`

Step 2: download and prepare the dataset

    download SceneFLow dataset: "FlyingThings3D", "Driving" and "Monkaa" (final pass and disparity files).
  
      -mv all training images (totallty 29 folders) into ${your dataset PATH}/frames_finalpass/TRAIN/
      -mv all corresponding disparity files (totallty 29 folders) into ${your dataset PATH}/disparity/TRAIN/
      -make sure the following 29 folders are included in the "${your dataset PATH}/disparity/TRAIN/" and "${your dataset PATH}/frames_finalpass/TRAIN/":
        
        15mm_focallength	35mm_focallength		A			 a_rain_of_stones_x2		B				C
        eating_camera2_x2	eating_naked_camera2_x2		eating_x2		 family_x2			flower_storm_augmented0_x2	flower_storm_augmented1_x2
        flower_storm_x2	funnyworld_augmented0_x2	funnyworld_augmented1_x2	funnyworld_camera2_augmented0_x2	funnyworld_camera2_augmented1_x2	funnyworld_camera2_x2
        funnyworld_x2	lonetree_augmented0_x2		lonetree_augmented1_x2		lonetree_difftex2_x2		  lonetree_difftex_x2		lonetree_winter_x2
        lonetree_x2		top_view_x2			treeflight_augmented0_x2	treeflight_augmented1_x2  	treeflight_x2	
	
    download and extract Carla, kitti and kitti2015 datasets.
        
Step 3: revise parameter settings and run "train.sh" and "predict.sh" for training, finetuning and prediction/testing. Note that the “crop_width” and “crop_height” must be multiple of 48 (for "DSMNet") or 64 (for "DSMNet2x2"), "max_disp" must be multiple of 12 (for "DSMNet") or 16 (for "DSMNet2x2") (default: 192).


## Pretrained models:

Updating ...


## Great Generalization Abilities:
DSMNet has great generalization abilities on other datasets/scenes.



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

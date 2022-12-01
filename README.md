# Optical Flow Estimation with Separable Cost Volume

Expansion of the 
[implementation](https://github.com/feihuzhang/SeparableFlow)
accompanying 
[Separable Flow: Learning Motion Cost Volumes for Optical Flow Estimation](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Separable_Flow_Learning_Motion_Cost_Volumes_for_Optical_Flow_Estimation_ICCV_2021_paper.pdf)
.

Additions:
* Implemented learnable, attention-based self-compression channels of the 3D correlation volume as described in the paper. The number of the
channels can be controlled via the command line.
* Implemented CUDA C++ PyTorch extension for cost volume separation that saves memory by not storing the 4D correlation volume.
* Option to omit 4D correlation volume aggregation.
* Option to omit the use of 4D correlation features during motion refinement.
* Option to add Global Motion Aggregation to Separable Flow.


## Building Requirements:

    gcc: >=5.3 and gcc != 10.3 (segfault in chrono during template resolution)
    gcc: 9.4.0 and 10.2 seem to work
    GPU mem: >=5G (for testing);  >=11G (for training)
    pytorch: >=1.6
    pytorch: python3.9 with pytorch 11.0 and cuda 11.3 works
    cuda: >=9.2 (9.0 doesn’t support well for the new pytorch version and may have “pybind11 errors”.)
    system-cuda: 11.2 works
    tested platform/settings:
      1) ubuntu 18.04 + cuda 11.0 + python 3.6, 3.7
      2) centos + cuda 11 + python 3.7
      

## Environment:

    
    pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    pip install matplotlib tensorboard scipy
    pip install einops opencv-python pypng


## How to Use?

Step 1: compile the libs by "sh compile.sh"
- Change the environmental variable ($PATH, $LD_LIBRARY_PATH etc.), if it's not set correctly in your system environment (e.g. .bashrc). Examples are included in "compile.sh".


Step 2: optionally, compile the alternative 4D correlation volume separation by "sh compile_memsave.sh"


Step 3: download and prepare the training dataset or your own test set.

        
Step 4: revise parameter settings and run "train.sh" and "evaluate.sh" for training, finetuning and prediction/testing. Note that the “crop_width” and “crop_height” must be multiple of 64 during training.

## Folders and Files

<ins>checkpoints</ins>  
Training configurations and corresponding model checkpoints.

<ins>core</ins>  
Python files for the separable flow and gma implementation.

<ins>libs/MemorySaver</ins>  
Alternative correlation volume separation implementation.

<ins>scripts</ins>  
Python programs and shell scripts for evaluation and testing.

## Notable Command Line Arguments for Training and Inference

Command line arguments for train.py and evaluate.py

Model modification settings:

      --no_4d_corr        whether to use 4d correlation features during motion refinement (default: use 4D correlation features)
      --num_corr_channels number of channels of the 3D correlation volume (default: 2)
      --no_4d_agg         whether to aggregate the 4D correlation volume (default: use 4D aggregation)

Use alternate implementation that does not store 4d correlation volume

      --alternate_corr            use alternate 4D correlation volume separation implementation for only the forward pass (inference, default: false)
      --alternate_corr_backward   use alternate correlation volume separation implementation supporting backward pass (training, default: false)


Global Motion Aggregation settings:

      --use_gma               whether to use Global Motion Aggregation (default: do not use GMA)
      --position_only         only use position-wise attention (default: content only)
      --position_and_content  use position and content-wise attention (default: content only)
      --num_heads             number of heads in attention and aggregation (default: 1)

Experiment (Multi-Training) settings:

      --run_name              name used to identify the current run of the script (e.g. chairs, things, etc.)
      --experiment_name       name used to identify the current experiment (e.g. no4dCorrelationFeatures)


## Experiment Structure

      Experiment Name (experiment root folder)
      > models   
        > chairs.pth
        > things.pth
      > runs
      > > chairs
      > > > log.txt
      > > > tensorboard: events.out.tfevents...
      > > things
      > > > log.txt
      > > > tensorboard: events.out.tfevents...
      > slurm
      > > job.sh
      > > run_me.sh
      > slurm_output
      > > job_*_output.txt
      > > job_*_error.txt
      > train.sh


* <ins>Folder</ins>: Experiment Name  
  named after experiment_name specified in train.sh
* <ins>train.sh</ins>:  
Contains the training schedule in the form of the training script with command line arguments  
python train.py --arg1 val1 --arg2 val2 ... --argN valN
* <ins>models</ins>:  
suggested path of the folder to store the models, can be set as command line argument in train.sh
* <ins>runs</ins>:  
contains folders with the --run_name and the logfiles corresponding to this execution of the script
* <ins>slurm</ins>:  
  * <ins>job.sh</ins>: specifies the slurm parameters and runs run_me.sh
  * <ins>run_me.sh</ins>: prints out initial information such as data and time and starts train.sh


## References:

* Separable Flow 
( [paper](
  https://github.com/feihuzhang/SeparableFlow) 
 | 
[implementation](
  https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Separable_Flow_Learning_Motion_Cost_Volumes_for_Optical_Flow_Estimation_ICCV_2021_paper.pdf) 
):  

        @inproceedings{Zhang2021SepFlow,
          title={Separable Flow: Learning Motion Cost Volumes for Optical Flow Estimation},
          author={Zhang, Feihu and Woodford, Oliver J. and Prisacariu, Victor Adrian and Torr, Philip H.S.},
          booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
          year={2021}
          pages={10807-10817}
        }

* RAFT 
(
[paper]()
 | 
[implementation](https://github.com/princeton-vl/RAFT)
)

      @inproceedings{teed2020raft,
        title={RAFT: Recurrent All Pairs Field Transforms for Optical Flow},
        author={Zachary Teed and Jia Deng},
        booktitle={Europe Conference on Computer Vision (ECCV)},
        year={2020}
      }

* GMA (
  [paper](https://arxiv.org/abs/2104.02409)
  | 
  [implementation](https://github.com/zacjiang/GMA)
)

      @inproceedings{jiang2021learning,
        title={Learning to estimate hidden motions with global motion aggregation},
        author={Jiang, Shihao and Campbell, Dylan and Lu, Yao and Li, Hongdong and Hartley, Richard},
        booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
        pages={9772--9781},
        year={2021}
      }
* GANet (
[paper](https://arxiv.org/pdf/1904.06587.pdf)
  | 
[implementation](https://github.com/feihuzhang/GANet)
)

      @inproceedings{Zhang2019GANet,
        title={GA-Net: Guided Aggregation Net for End-to-end Stereo Matching},
        author={Zhang, Feihu and Prisacariu, Victor and Yang, Ruigang and Torr, Philip HS},
        booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        pages={185--194},
        year={2019}
      }
* DSMNet
( [paper](
https://arxiv.org/pdf/1911.13287.pdf
)
 | 
 [implementation](
https://github.com/feihuzhang/DSMNet
 ) )

      @inproceedings{zhang2019domaininvariant,
        title={Domain-invariant Stereo Matching Networks},
        author={Feihu Zhang and Xiaojuan Qi and Ruigang Yang and Victor Prisacariu and Benjamin Wah and Philip Torr},
        booktitle={Europe Conference on Computer Vision (ECCV)},
        year={2020}
      }

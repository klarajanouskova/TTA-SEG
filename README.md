## Test-Time Adaptation foMasked Autoencoders 

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>


Based on official re-implementation of the paper [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377):


### Software
The environment the repo has been running on can be found in the file env_tta.yml file. Unfortunately, the specific nightly versions of torchvision etc the repo relies on can't be installed from the conda file - I think the best approach might be to install those separately, remove them from the conda environment file and install the rest of the packages from the conda .yml file. 

Using the nightly torch versions allows us to use the new torchvision transforms API and thus simplify the code. Also, it would be nice to try out the compile functionality to potentially speed up the training.

### Code structure
The repository is split into two parts. The root folder contains the experiments with the 
Class-Agnostic Semantic-Segmentation Model (CASS) and is described in the rest of this
README. The SAM subfolder contains the experiments with the SegmentAnything Model and
have the same structure.

### Training

All the training is done on top of the pretrained ViT-B model (the smallest one), download instructions can be found here (original repo readme): [https://github.com/klarajanouskova/TTA#fine-tuning-with-pre-trained-checkpoints](https://github.com/klarajanouskova/TTA#fine-tuning-with-pre-trained-checkpoints)

The training can be run by launching the scripts starting with the `main_` prefix,
which contain model and dataset loading. The code for each train and validation epoch is then
contained in the files starting wtih the `engine_` prefix.

The following scripts have beeen tested recently and should work, others may need modification to
work with the latest version of the code:

* `main_finetune_seg.py` - joint finetuning of both segmentation and reconstruction, currently on the pascal VOC dataset
* `main_train_seg_loss.py` - training of the deep  (at test time) self-supervised segmentation loss.

The code is meant to be run in distributed mode, an example of segmentation training on 2 GPUs
on a single node:

```
python -m torch.distributed.launch --nproc_per_node=2 main_finetune_seg.py --world_size 2
```

### Sweeps

It is recommended to use WandB sweeps to run hyperparameter sweeps.

You can initialize it from a config file, e.g.:

```
wandb sweep config/seg_ft_config.yml
```

and then run the sweep with:

```
wandb agent <sweep_id>
```

Note that you can launch multiple agents on different nodes once 
the sweep was initialized.

For more information about sweeps, please check the [WandB documentation](https://docs.wandb.com/sweeps).

### TTA
The main logic of the TTA method is in the `tta.py`
file - there is a a wrapper TestTimeAdaptor class. 
To run hyper-parameter sweeps on VOC-C, look at the `inference_tta.py` file.

### Corruptions

Corruption implementation can be found in the `distortion.py` file.

# Explanation of all config attributes
The configs follow a hierarchical style as used by [hydra](https://hydra.cc/). E.g. as part of the ```general``` section we can modify the ```force_cpu``` attribute in the config in yaml style as
```
general:
  force_cpu: True
```
or on the command line via ```general.cpu=True```

## Base sections
| Attribute Name | Description | Link to Section |
| --- | --- | --- |
| general | General attributes of the training. | [general](#general) |
| data | All attributes defining the data. | [data](#data) |
| loader | Attributes regarding data loader. | [loader](#loader)
| model | Attributes defining the model. | [model](#model) |
| transforms | Attributes defining data transforms. | [transforms](#transforms) |
|hyperparams | Defining hyperparameters of training | [hyperparams](#hyperparams) |
|wandb | Setting options for weights & biases. | [wandb](#wandb) |



## general
| Attribute Name | Description | Default Value |
| --- | --- | --- |
| force_cpu | Train only on the CPU, no GPU support. | False |
| log_wandb | Use weights&biases for logging training curves | False |
| seed | seed for pseudorandom number generator | 0 |
| compile | Use ```torch.compile``` feature | False |
| log_images | log images to weights&biases | False |
| private_data | Prevent logging attention maps to weights&biases, which could leak private data to an external server | True |
| output_save_folder | Path to log model weights and figures | None |

## data
| Attribute Name | Description | Default Value | Options |
| --- | --- | --- | --- |
| name | Name of dataset type | | ```ct_imagefolder```, ```organmnist3d```, ```nodulemnist3d```, ```fracturemnist3d```, ```adrenalmnist3d```, ```vesselmnist3d```, ```synapsemnist3d``` |
| base_path | Path to dataset (only for ct_imagefolder) | | |
| loader | Use DataLoader from [torch](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) or [MonAI](https://docs.monai.io/en/stable/data.html#dataloader) library | torch | ```torch```, ```monai``` |

## loader
Any keyword here is directly passed as argument to the initialization of the dataloader. Examples could be ```loader.num_workers = 16```. For a detailed list of options please look at [torch DataLoader class](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) or [monai DataLoader](https://docs.monai.io/en/stable/data.html#dataloader)

## transforms
| Attribute Name | Description | Default Value | Options |
| --- | --- | --- | --- |
| tf_library | Library to use for transforms | [torchio](https://torchio.readthedocs.io/) | torchio, monai |
| train_tf | All train transforms. Names correspond to names from transform library, arguments are passed to transforms. See example below | |
| val_tf | All validation transforms. Names correspond to names from transform library, arguments are passed to transforms. See example below | |
| test_tf | All test transforms. Names correspond to names from transform library, arguments are passed to transforms. See example below | |

Example transform config:
```
transforms:
  train_tf:
    RandomFlip:
      flip_probability: 0
```

## model
| Attribute Name | Description | Default Value | Options |
| --- | --- | --- | --- |
| type | Model conversion strategy | None | ```twop5_att```, ```twop5_pool```, ```twop5_lstm```, ```twop5_tf```, ```acs_direct```, ```acs_3d```, ```acs_twop5```. |
| data_view_axis | From which side to disassemble 3D data. | all_sides | ```all_sides```, ```only_x```, ```only_y```, ```only_z```. |
| backbone | Backbone to use as model architecture | None | ```resnetxx```, where ```xx``` in [18, 34, 50, 101, 152], ```vit_xxxx```, where ```xxxx``` in [b_16, b_32, l_16, l_32, h_14] |
| in_channels | Number of channels of input data | | Any integer |
| num_classes | Number of classes of dataset | | Any integer |
| feature_dim | Number of features in intermediate representation. | Standard value of architecture | Any integer |
| unfreeze | Gradually unfreeze model during training. See [UnfreezeConfig](#unfreezeconfig) | | |
| additional_args | Arbitrary additional keywords for model class | | |

### UnfreezeConfig
| Attribute Name | Description | Default Value | Options |
| --- | --- | --- | --- |
| train_mode | Switch model from eval to train mode (activate dropout and batchnorm). | -1 (from beginning) | Any integer (epoch). |
| feature_extractor | Set for modules of feature extractor ```requires_grad=True``` | -1 (from beginning) | Any integer (epoch). |

## hyperparams
| Attribute Name | Description | Default Value | Options |
| --- | --- | --- | --- |
| epoch | Number of epochs to train | None | Any integer |
| train_bs | Training batch size | None | Any integer |
| val_bs | Validation batch size | None | Any integer |
| test_bs | Test batch size | None | Any integer |
| grad_acc_steps | Steps to accumulate gradient before doing update step. Artificially increases batch size, e.g., when VRAM is limited. | 1 | Any integer |
| overfit | Set number of batches to overfit to | None (no overfitting) | Any integer |
| opt_args | Additional arguments for optimizer, e.g., learning rate. Directly passed to [optimizer class](https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html). | None | |

## wandb
Any keyword here is directly passed as argument to the initialization of the wandb. Examples could be ```wandb.project = twoandahalfdimensions```. For a detailed list of options please look at [wandb](https://docs.wandb.ai/ref/python/init).
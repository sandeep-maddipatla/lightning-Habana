Repository for DETR ResNet-50 enabling and tuning

## Required Repo Checkouts

- This Repository and branch
  - Clone to ~/lightning-Habana
- Optimum Habana for model patches to workaround problems encountered with this model and to optimize performance.
  - Currently these changes are in [this pull request](https://github.com/huggingface/optimum-habana/pull/1334)
  - Checkout the detr-hpu branch to ~/optimum-habana
  - When this change is merged as part of optimum-habana release, these steps related to optimum-habana checkout are unnecessary

        git clone https://github.com/huggingface/optimum-habana.git && cd optimum-habana
        gh pr checkout 1334 && cd ..

## Dataset checkout

This test uses the [CPPE-5 dataset](https://huggingface.co/datasets/rishitdagli/cppe-5). 

The dataset should be downloaded to `~/CPPE-Dataset`. It is recommended to follow download instructions from [the maintainer's Github](https://github.com/Rishit-dagli/CPPE-Dataset)

Please edit `init_datasets` and `prepare_dataloader` functions appropriately if a different dataset is used.

## HPU Execution

Launch docker:

* Version 1.17

      docker run -it --rm --name test_1.17 --runtime=habana -e HABANA_VISIBLE_DEVICES=all
                 -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e http_proxy=http://proxy-dmz.intel.com:912
                 -e https_proxy=http://proxy-dmz.intel.com:912
                 --cap-add=sys_nice --net=host --ipc=host
                 -v $HOME:/root --workdir /root
                  vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:1.17.0-495

Note that above launch mounts the $HOME directory of the user to /root in the docker container

* In the container, issue following commands to prepare env, if working off a unmerged branch/PR to optimum-habana 

      cd ~/lightning-Habana/examples/pytorch/detr
      pip install -r requirements.txt

      # Below required for working off a PR/ unmerged branch of optimum-habana
      cd ~/optimum-habana
      pip uninstall -y optimum-habana
      python setup.py build
      python setup.py install
      cd ~/lightning-Habana/examples/pytorch/detr

  * If working off a stable optimum-habana that contains all required changes, only the first two steps are required
    * Optimum Habana comes installed with the docker container, and no other changes are needed then.

### Recommended Workflow

Below to be executed within the docker container launched with above step.

#### Running Training:

It is recommended to use lazy mode with `--pad` and `--num-buckets 1` for performance. Sample commmand line below:

    python detr_ft.py  --max-epochs 10 --pad --num-buckets 1 2>&1 | tee console.log

Note:
- The model patches in optimum-habana are meant to have the model ignore padded objects and
  generate quality equivalent to `num-buckets 0`. Of course, `--num-buckets 1` is still required for
  performance to avoid recompilations in lazy mode execution

#### Running Inference:
 
The inference script `detr_inference.py` is designed to run inference on all images in the validation dataset and
save annotated output images to the disk.

It accepts an argument specifying the checkpoint file to be used in running the inference. This is achieved
by specifying `--use-ckpt --ckt-path /path/to/checkpoint` on the inference script invocation. Sample below:

    python detr_inference.py --pad --num-buckets 0 --use-ckpt --ckpt-path hpu_training_10epochs_ob1/cppe-5.ckpt 2>&1 | tee console.log

### Other Commonly Used Training Command lines

The default execution leverages the lazy mode execution for Gaudi. Sometimes, for debug/other reasons, we may prefer to use eager mode execution,
and have the code execution in a deterministic way with randomness associated with models and parallel execution eliminated.

Here are command lines for some such common cases:

  * Lazy Mode, Deterministic

        PT_HPU_LAZY_MODE=1 python detr_ft.py --max-epochs 5 --pad --num-buckets 1 --deterministic 2>&1 | tee console.log

  * Eager Mode, Deterministic

    In example below, padding and object bucketing are disabled.

        PT_HPU_LAZY_MODE=0 python detr_ft.py --max-epochs 5 --no-pad --num-buckets 0 --deterministic 2>&1 | tee console.log

  * Lazy Mode, Non-deterministic with checkpoints dumped at 5-epoch intervals

        PT_HPU_LAZY_MODE=1 python detr_ft.py --max-epochs 25 --pad --num-buckets 1 --ckpt-store-interval-epochs 5 2>&1 | tee console.log

  * Add `--autocast` to enable autocast and run in mixed precision mode using BFloat16.

        PT_HPU_LAZY_MODE=1 python detr_ft.py  --max-epochs 10 --pad --num-buckets 1 --autocast 2>&1 | tee console.log 
 
    Note: Not all combinations of input settings have been tested. 

## Other Modes of Execution

The scripts provide token support for CPU and CUDA device execution, but these modes aren't well-tested.


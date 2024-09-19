Repository for DETR ResNet-50 enabling and tuning

## Required Repo Checkouts

- This Repository and branch
  - Checkout to ~/gs-274/detr-ft
- Optimum Habana for model patches to workaround problems encountered with this model and to optimize performance.
  - Currently these changes are in [this fork](https://github.com/sandeep-maddipatla/optimum-habana.git)
  - Checkout the detr-hpu branch to ~/optimum-habana
  - When this branch is merged as part of optimum-habana release, these steps related to optimum-habana are unnecessary

        git clone https://github.com/sandeep-maddipatla/optimum-habana.git && cd optimum-habana
        git checkout detr-hpu && cd ..

## Dataset checkout

This test uses the [CPPE-5 dataset](https://huggingface.co/datasets/rishitdagli/cppe-5). Download with following steps

    cd ~/gs-274/detr-ft
    bash prepare_dataset.sh

The dataset should be downloaded to `~/gs-274/CPPE-Dataset`

## HPU

Launch docker:

* 1.17

      docker run -it --rm --name sandeep_1.17 --runtime=habana -e HABANA_VISIBLE_DEVICES=all
                 -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e http_proxy=http://proxy-dmz.intel.com:912
                 -e https_proxy=http://proxy-dmz.intel.com:912
                 --cap-add=sys_nice --net=host --ipc=host
                 -v $HOME:/root --workdir /root
                  vault.habana.ai/gaudi-docker/1.17.0/ubuntu22.04/habanalabs/pytorch-installer-2.3.1:1.17.0-495

* In the container, issue following commands to prepare env, if working off a unmerged branch/PR to optimum-habana 

      cd ~/gs-274/detr-ft
      pip install -r requirements.txt

      # Below required for working off a PR/ unmerged branch of optimum-habana
      cd ~/optimum-habana
      pip uninstall -y optimum-habana
      python setup.py build
      python setup.py install
      cd ~/gs-274/detr-ft

  * If working off a stable optimum-habana that contains all required changes, only the first two steps are required
    * Optimum Habana comes installed with the docker container, and no other changes are needed then.

### Recommended Workflow

Below to be executed within the docker container launched with above step.

#### Running Training:

Recommended to use lazy mode with `--pad` and `--num-buckets 1` for performance. Sample commmand line below:

    PT_HPU_LAZY_MODE=1 PT_HPU_METRICS_FILE=~/metricslog.json PT_HPU_METRICS_DUMP_TRIGGERS=process_exit,metric_change  OUTDIR=hpu_training_100epochs_ob1 OPTIONS="--max-epochs 100 --pad --num-buckets 1 --ckpt-store-interval-epochs 5" ./run.sh

Note:
- The option `--ckpt-store-interval-epochs 5` dumps the checkpoint for every few epochs that can
  allow assessment of model quality in post. This step can potentially be integrated into the validation
  step of the model, but we have chosen to keep it separate for now.
  
- The model patches in optimum-habana are meant to have the model ignore padded objects and
  generate quality equivalent to `num-buckets 0`. Of course, `--num-buckets 1` is still required for
  performance to avoid recompilations in lazy mode execution

#### Inference and Quality assessment:
 
The inference script `detr-inference.py` is designed to run inference on all images in the validation dataset and
log mAP scores on the console. The `run_inference.sh` script is a wrapper over this inference script similar to
`run.sh` for training.

We loop over all checkpoints generated in training to get mAP scores as a function of epochs run. This is achieved
by specifying `--use-ckpt --ckt-path /path/to/checkpoint` on the inference script invocation. Sample below:

    for x in $(ls /path/to/training/outdir/*.ckpt); do PT_HPU_LAZY_MODE=1 PT_HPU_METRICS_FILE=~/metricslog.json PT_HPU_METRICS_DUMP_TRIGGERS=process_exit,metric_change  OUTDIR=hpu_inference_100epochs/$(basename $x) OPTIONS="--pad --num-buckets 0 --use-ckpt --ckpt-path $x" ./run_inference.sh; done

Simply look for `metrics` in the console log for a summary of the quality metrics.

    grep metrics /path/to/inference_loop_outdir/*/console.log


### Other Commonly Used Training Command lines

  * Lazy Mode, Deterministic
    
        PT_HPU_LAZY_MODE=1 PT_HPU_METRICS_FILE=~/metricslog.json PT_HPU_METRICS_DUMP_TRIGGERS=process_exit,metric_change  OUTDIR=hpu_training_5epochs_deterministic OPTIONS="--max-epochs 5 --pad --num-buckets 1 --deterministic" ./run.sh

  * Eager Mode, Non-Deterministic
    In example below, padding and object bucketing are disabled.

        PT_HPU_LAZY_MODE=0 PT_HPU_METRICS_FILE=~/metricslog.json PT_HPU_METRICS_DUMP_TRIGGERS=process_exit,metric_change  OUTDIR=hpu_training_5epochs OPTIONS="--max-epochs 5 --no-pad --num-buckets 0" ./run.sh

  * Lazy Mode, with checkpoints dumped at intervals

        PT_HPU_LAZY_MODE=1 PT_HPU_METRICS_FILE=~/metricslog.json PT_HPU_METRICS_DUMP_TRIGGERS=process_exit,metric_change  OUTDIR=hpu_training_100epochs_ob3 OPTIONS="--max-epochs 100 --pad --num-buckets 3 --ckpt-store-interval-epochs 5" ./run.sh

  * Inference using Checkpoints generated from training.

        PT_HPU_LAZY_MODE=1 PT_HPU_METRICS_FILE=~/metricslog.json PT_HPU_METRICS_DUMP_TRIGGERS=process_exit,metric_change  OUTDIR=hpu_inf_5epochs_ob2 OPTIONS="--use-ckpt --ckpt-path hpu_training_100epochs_ob3/cppe-5.ckpt" ./run_inference.sh

    * Dumps metrics on quality, and annotated output images
    * Runs through validation dataset
 

  * Add `--bf16` to run in bf16 precision

  * Add `--autocast` to enable autocast.
 
    Note: Not all combinations of input settings have been tested. 

## CUDA

Note: Not well-tested.

At least one known issue: Fails to work if  `--deterministic` is specified. 

* Launch docker:

`docker run --gpus all -it --rm --shm-size=20g --ulimit memlock=-1 -v ~/sandeep:/root nvcr.io/nvidia/pytorch:24.07-py3`

* Run within container:

      # Prepare env
      cd ~/gs-274/detr-ft/
      pip install -r requirements.txt
      pip install lightning torchmetrics

      # Training
      OUTDIR=cuda_pad_5epochs OPTIONS="--device cuda --max-epochs 5 --max-steps 10000 --pad" ./run_cuda.sh

      # Inference with weights from training
      OUTDIR=cuda_pad_5epochs_inference OPTIONS="--device cuda --use-ckpt --ckpt-path cuda_pad_5epochs/cppe-5.ckpt" ./run_inference_cuda.sh

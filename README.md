# Supplementary code repository for NeurIPS2023 submission "Masked Autoencoders with Multi-Window Attention Are Better Audio Learners"

## Contents
* [Pre-trained weights corresponding to default MAE and MW-MAE configurations](pretrained_weights)
* Our local copy of [hear-eval-kit](hear-eval-kit) for easy downstream reproducibility. Original can be found [here](https://github.com/hearbenchmark/hear-eval-kit)
* [Feature extraction API](mwmae/hear_api) compatible with the [hear-eval-kit](https://github.com/hearbenchmark/hear-eval-kit) format for extracting features.
* Helper code to [extract features](mwmae/extract_features.sh) and [run downstream experiments](mwmae/downstream_experiments.sh) on provided pre-trained models

## Setup
* Required: cuda 11.x, cudnn 8.2 or newer.
* create a new conda environment with python 3.9
* Setting up `jax==0.3.4` & `jaxlib==0.3.2+cuda11.cudnn82`

Follow these steps
```shell
conda create -n mwmae-env python=3.9 -y
conda activate mwmae-env

# install pre jax requirments
pip install -r pre_jax_requirements.txt

# install hear-eval-kit specific requirements
pip install -r hear-eval-kit/requirements.txt

# install hear-eval-kit, WITHOUT AUTO DEPS
cd hear-eval-kit && pip install --no-deps . && cd ..

# install jax
pip install --upgrade "jax[cuda]==0.3.4" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# install flax and other jax requirements
pip install -r post_jax_requirements.txt

```

## Get 16000 Hz data from hear
* Follow https://hearbenchmark.com/hear-tasks.html to get data. By default, data on HEAR's zenodo page is 48000 Hz.
* We recommend downloading data directly from HEAR's [GCS bucket](gs://hear2021-archive/tasks/), where you can find preprocessed 16000 Hz data.
* Extract all the files to a folder `$TASKS_DIR`

## Get pretrained weights

* Pre-trained can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1NVEAOXsCNbpwhuH5zsyolAQm4xPggFiH?usp=sharing)
* Download the entire `pretrained_weights` folder in the root repository, and paths will work automatically.

## Extract features

```shell
cd mwmae

./extract_features.sh $TASKS_DIR $OUTPUT_DIR
```
where TASKS_DIR is the directory where you extracted tasks from HEAR-2021 to, and OUTPUT_DIR is the base directory where output features will be stored.
This also prepares a `todo_audioset` directory in OUTPUT_DIR, which is setting up for downstream classification on 10 seeds.

## Run downstream experiments

```shell
cd mwmae

./downstream_experiments.sh mae_200_16x4_384d_8h_4l $OUTPUT_DIR
./downstream_experiments.sh mw_mae_200_16x4_384d_8h_4l $OUTPUT_DIR
```

This will run downstream experiments on all the extracted features on 10 random seeds.

## Get results
Finally, you can run the following script to get results of downstream experiments of the two models

```shell
python stats_aggregation.py --base_dir ${OUTPUT_DIR}/todo_audioset --output_dir ${OUTPUT_DIR}/parsed_results
```

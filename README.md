# Bridge-Prompt: Towards Ordinal Action Understanding in Instructional Videos

Created by [Muheng Li](https://ttlmh.github.io/), [Lei Chen](http://ivg.au.tsinghua.edu.cn/people/Lei_Chen/), [Yueqi Duan](https://duanyueqi.github.io/), Zhilan Hu, [Jianjiang Feng](https://scholar.google.com/citations?user=qlcjuzcAAAAJ&hl=en), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=en&authuser=1)

This repository contains PyTorch implementation for Bridge-Prompt (CVPR 2022).

We propose a prompt-based framework, **Bridge-Prompt (Br-Prompt)**, to model the semantics across multiple adjacent correlated actions, so that it simultaneously exploits both out-of-context and contextual information from a series of ordinal actions in instructional videos. More specifically, we reformulate the individual action labels as integrated text prompts for supervision, which bridge the gap between individual action semantics. The generated text prompts are paired with corresponding video clips, and together co-train the text encoder and the video encoder via a contrastive approach. The learned vision encoder has a stronger capability for ordinal-action-related downstream tasks, e.g. action segmentation and human activity recognition.

![intro](pipeline.gif)

Our code is based on [CLIP](https://github.com/openai/CLIP) and [ActionCLIP](https://github.com/sallymmx/ActionCLIP).

## Prerequisites

### Requirements

- [PyTorch](https://pytorch.org/) >= 1.8
- [wandb](https://wandb.ai/)
- dotmap
- yaml
- pprint
- tqdm
- RandAugment

You may need [ffmpeg](https://www.ffmpeg.org/) for video data pre-processing.

The environment is also recorded in *requirements.txt*, which can be reproduced by

```
pip install -r requirements.txt
```

## Pretrained models

We use the base model (ViT-B/16 for image encoder & text encoder) pre-trained by [ActionCLIP](https://github.com/sallymmx/ActionCLIP) based on Kinetics-400. The model can be downloaded in [link](https://pan.baidu.com/s/1Gdz8f1AwBKcbX61-qI2qxQ) (pwd:ilgw). The pre-trained model should be saved in ./models/.

## Datasets

Raw video files are needed to train our framework. Please download the datasets with RGB videos from the official websites ( [Breakfast](https://serre-lab.clps.brown.edu/resource/breakfast-actions-dataset/) / [GTEA](https://cbs.ic.gatech.edu/fpv/) / [50Salads](https://cvip.computing.dundee.ac.uk/datasets/foodpreparation/50salads/) ) and save them under the folder ./data/(name_dataset). For convenience, we have used the extracted frames of the raw RGB videos as inputs. You can extract the frames from raw RGB datasets by running:

```
python preprocess/get_frames.py --dataset (name_dataset) --vpath (folder_to_your_videos) --fpath ./data/(name_dataset)/frames/
```

To be noticed, [ffmpeg](https://www.ffmpeg.org/) is needed here for frame extraction.

Furthermore, please also extract the .zip files to ./data/(name_dataset) respectively.

## Training

- To train Bridge-Prompt on Breakfast from Kinetics400 pretrained models, you can run:
  
```
bash scripts/run_train.sh  ./configs/breakfast/breakfast_ft.yaml
 ```

- To train Bridge-Prompt on GTEA from Kinetics400 pretrained models, you can run:

```
bash scripts/run_train.sh  ./configs/gtea/gtea_ft.yaml
```

- To train Bridge-Prompt on 50Salads from Kinetics400 pretrained models, you can run:

```
bash scripts/run_train.sh  ./configs/salads/salads_ft.yaml
```

- **We have uploaded the trained weights of our model on several datasets.** 
  
  For *action segmentation*, you can download from here:
  
  | dataset   | split | checkpoint                                                       |
  |:--------- |:----- |:---------------------------------------------------------------- |
  | GTEA      | 1     | [link](https://pan.baidu.com/s/1KlRtqUuAEO8FRiKnBQW3NA?pwd=fbj9) |
  | GTEA      | 2     | [link](https://pan.baidu.com/s/1PG6lgdHxLTEJW7JFRHkljQ?pwd=1hdc) |
  | GTEA      | 3     | [link](https://pan.baidu.com/s/12O45DcI19rTrmhtpIjjLfg?pwd=ocso) |
  | GTEA      | 4     | [link](https://pan.baidu.com/s/1F8V83dAWFXJpR1giRMQO2g?pwd=ztnv) |
  | 50 Salads | 1     | [link](https://pan.baidu.com/s/1PVO8jpU5WvCtgS68thB38w?pwd=6uqb) |
  | 50 Salads | 2     | [link](https://pan.baidu.com/s/1lSIYlkcfAe30sETC5iacjQ?pwd=0glk) |
  | 50 Salads | 3     | [link](https://pan.baidu.com/s/13PL8rlIir6j4ni2b4Mh0Hw?pwd=14i8) |
  | 50 Salads | 4     | [link](https://pan.baidu.com/s/1ePUhADbTzgdMfvcsMGPxpg?pwd=8z0t) |
  | 50 Salads | 5     | [link](https://pan.baidu.com/s/1wqeh1hjVba_w_p7_kPjQMA?pwd=6p1h) |
  
  For *activity recognition*, you can download the trained weights on Breakfast from this [link](https://pan.baidu.com/s/1xHTIvMwBfL-3HS4S5esxfQ?pwd=0pe5).

## Extracting frame features

We use the Bridge-Prompt pre-trained image encoders to extract frame-wise features for further downstream tasks (*e.g.* action segmentation). You can run the following command for each dataset respectively:

```
python extract_frame_features.py --config ./configs/(dataset_name)/(dataset_name)_exfm.yaml --dataset (dataset_name)
```

Since 50Salads/Breakfast are large scale datasets, we extract the frame features by window splits. To combine the splits, please run the following command:

```
python preprocess/combine_features.py
```

Please modify the variables *dataset* and *feat_name* in combine_features.py for each dataset.

## Action segmentation

You can reproduce the action segmentation results using [ASFormer](https://github.com/ChinaYi/ASFormer) by the previously extracted frame features.

## Activity recognition

You can reproduce the activity recognition results using the command:

```
python ft_acti.py
```

based on the previously extracted frame features (Breakfast).

## Ordinal action recognition

The ordinal action inferences are executed using the command:

```
bash scripts/run_test.sh  ./configs/(dataset_name)/(dataset_name)_test.yaml
```

and check the accuracies using:

```
bash preprocess/checknpy.py
```

Please modify the variables *dataset* in checknpy.py for each dataset.

## Notes

Please modify *pretrain* in all config files according to your own working directions.

## License

MIT License.

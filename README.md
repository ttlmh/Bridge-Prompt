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

Please download the datasets(Breakfast/GTEA/50Salads) from [link](https://zenodo.org/record/3625992#.YZ5tAdBBxPZ) provided by [MS-TCN++](https://github.com/sj-li/MS-TCN2). We extract the frames from raw videos using ffmpeg:

```
cmd = "ffmpeg -i " + videopath + " -vsync vfr " + videoname + "/img_%05d.jpg"
```

Furthermore, please extract the .zip files to ./data/(name_dataset) respectively.



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


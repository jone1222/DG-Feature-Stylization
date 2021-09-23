# Feature Stylization and Domain-aware Contrastive Loss for Domain Generalization

### This is an official implementation of "Feature Stylization and Domain-aware Contrastive Loss for Domain Generalization" (ACMMM 2021 Oral)

> ## Feature Stylization and Domain-aware Contrastive Loss for Domain Generalization
> 
> Seogkyu Jeon, Kibeom Hong, Pilhyeon Lee, Jewook Lee, Hyeran Byun (Yonsei Univ.)
>
> Paper : [https://arxiv.org/abs/2108.08596](https://arxiv.org/abs/2108.08596)
> 
> **Abstract**: Domain generalization aims to enhance the model robustness against domain shift without accessing the target domain. Since the available source domains for training are limited, recent approaches focus on generating samples of novel domains. Nevertheless, they either struggle with the optimization problem when synthesizing abundant domains or cause the distortion of class semantics. To these ends, we propose a novel domain generalization framework where feature statistics are utilized for stylizing original features to ones with novel domain properties. To preserve class information during stylization, we first decompose features into high and low frequency components. Afterward, we stylize the low frequency components with the novel domain styles sampled from the manipulated statistics, while preserving the shape cues in high frequency ones. As the final step, we re-merge both the components to synthesize novel domain features. To enhance domain robustness, we utilize the stylized features to maintain the model consistency in terms of features as well as outputs. We achieve the feature consistency with the proposed domain-aware supervised contrastive loss, which ensures domain invariance while increasing class discriminability. Experimental results demonstrate the effectiveness of the proposed feature stylization and the domain-aware contrastive loss. Through quantitative comparisons, we verify the lead of our method upon existing state-of-the-art methods on two benchmarks, PACS and Office-Home.

## Prerequisites

##### * The code is built upon popular DG pytorch toolbox [DASSL](https://github.com/KaiyangZhou/Dassl.pytorch).

### Dependency
- Python 3.6
- CUDA
- Pytorch 1.7
- Check the requirements.txt

```
pip install -r requirements.txt
```

### Installation

```bash
# Clone this repo
git clone https://github.com/jone1222/DG-Feature-Stylization
cd DG-Feature-Stylization/

# Create a conda environment
conda create -n featstyle python=3.7

# Activate the environment
conda activate featstyle

# Install dependencies
pip install -r requirements.txt

# Install torch (version >= 1.7.1) and torchvision
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
```

### Datasets
Download links of datasets are available in [DATASETS.md](DATASETS.md).

Please ensure that the downloaded datasets are located under the same root directory as follows:
```
dataset_root/
    pacs/
        images/
        splits/
    office_home_dg/
        art/
        clipart/
        product/
        real_world/
```

#### Training
##### PACS
```
bash train_pacs.sh
```

#### Inference
##### PACS

The model weights pre-trained on PACS can be downloaded [here](https://drive.google.com/file/d/1YAfwFxsvxl6MbR0RJ13bisZ4aUjJb1ah/view?usp=sharing).

```
bash test_pacs.sh
```

## Citation
If you find this work useful for your research, please cite:

```
@inproceedings{jeon2021stylizationDG,
  title={Feature Stylization and Domain-aware Contrastive Learning for Domain Generalization},
  author={Seogkyu Jeon and Kibeom Hong and Pilhyeon Lee and Jewook Lee and Hyeran Byun},
  booktitle={The 29th ACM International Conference on Multimedia},
  year={2021},
}
```

## Contact

For any comments or questions, please contact us via this email:
[jone9312@yonsei.ac.kr](jone9312@yonsei.ac.kr)

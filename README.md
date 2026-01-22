# Adapt Instead of Retrain: Temporal Adaptation via Memory Injection into Image Backbones for Low-Light Video Enhancement

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c?logo=pytorch&logoColor=white)

[![Image Crop Comparer](https://img.shields.io/badge/ImageViewer-Toolkit-orange)](https://github.com/Yangzhichen763/ImageCropComparator)

</div>

## TODO ✅

* [x] Release the official implementation of TAMI, including network code. This is a relatively rough version, so you may need some time to configure the environment and paths.

* [ ] Release pre-trained weights, training and inference configuration, training and inference scripts for reproducibility.

* [ ] Refactor and document code for clarity and reproducibility.

## Abstract 📌

Recent low-light image enhancement (LLIE) models have achieved impressive results on static inputs. However, when applied directly to dynamic video scenes, they often suffer from flickering and inconsistent illumination due to the absence of temporal modeling. To address this issue, we propose TAMI, a temporal adapter via memory injection, which enables powerful LLIE backbones to perform low-light video enhancement (LLVE) without full retraining. TAMI introduces a dual-path memory that stores paired temporal features from previous frames. It then uses an illumination-conditioned memory filter (ICMem) to retrieve the frame-relevant illumination prior from memory, which is then injected into the current frame through an illumination-guided feature enhancer (IGFE). This injection enables temporal adaptation by transferring prior enhancement knowledge into the backbone while preserving its spatial reasoning. To further stabilize illumination continuity and semantic alignment, we design an illumination map loss, a symmetric matching loss, and an intra- and inter-frame illumination alignment loss. Extensive experiments on video benchmarks demonstrate that TAMI substantially reduces temporal flickering and achieves a better trade-off between spatial fidelity and temporal coherence compared to existing LLVE baselines.

## Overview 🧩

![Overview of LLVE framework wit our proposed TAMI](figures/Overview%20Framework.png)

## Main Results 📊

Results are measured using `visualization/timeslice.py`. It should be noted that all metrics in our method are computed in the sRGB space, and no GT Mean-related techniques are applied.

For a detailed usage guide of the visualization tool, see [visualization/README.md](visualization/README.md).

<details>
<summary>Quantitative results on SDSD-indoor</summary>

![Quantitative comparison on SDSD-indoor](figures/Main%20Result%20Table1.png)

</details>


<details open>
<summary>Qualitative results on SDSD-indoor</summary>

![Qualitative comparison on SDSD-indoor](figures/Main%20Result%20Figure7.png)

</details>


<details>
<summary>Quantitative results on SDSD-outdoor</summary>

![Quantitative comparison on SDSD-outdoor](figures/Main%20Result%20Table2.png)

</details>


<details open>
<summary>Qualitative results on SDSD-outdoor</summary>

![Qualitative comparison on SDSD-outdoor](figures/Main%20Result%20Figure8.png)

</details>


To ensure fairness, if a comparison method does not provide pretrained weights, we retrain it using the recommended settings provided by the authors. 
Otherwise, we use the officially released pretrained weights for evaluation. 
All results are evaluated using a unified script, `visualization/timeslice.py`. In this paper, the following methods were retrained: Restormer, SNRNet, LLFormer, BiFormer, and HVI-CIDNet. The corresponding visual comparison results will be released later.

## Environment Setup 🧰

### 1) Create and Activate a Conda Environment

```bash
conda create --name TAMI python=3.10
conda activate TAMI
```

### 2) Install PyTorch

<details>
<summary>cuda-10.2</summary>

```bash
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
```

</details>

<details open>
<summary>cuda-11.6 (recommended)</summary>

```bash
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
```

</details>

<details>
<summary>cuda-12.1</summary>

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

</details>


### 3) Install Dependencies

```bash
pip install -r requirements.txt
```

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/Yangzhichen763/TAMI/blob/main/LICENSE) file.

## Acknowledgement

TAMI is built with reference to the code of the following projects: [BasicSR](https://github.com/XPixelGroup/BasicSR), [Restormer](https://github.com/swz30/Restormer), [XMem](https://github.com/hkchengrex/XMem) and [FastLLVE](https://github.com/Wenhao-Li-777/FastLLVE). Thanks for their awesome work!

The README is built with reference to [URWKV](https://github.com/FZU-N/URWKV). Thanks for their awesome work!

## Call to Action ⭐

* If you find this repo helpful, please consider starring it.
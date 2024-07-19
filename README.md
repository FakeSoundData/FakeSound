# FakeSound：Deepfake General Audio Detection
[![arXiv](https://img.shields.io/badge/arXiv-2406.08052-brightgreen.svg?style=flat-square)](https://arxiv.org/abs/2406.08052)
[![githubio](https://img.shields.io/badge/GitHub.io-Audio_Samples-blue?logo=Github&style=flat-square)](https://fakesounddata.github.io/)
[![Hugging Face data](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/ZeyuXie/FakeSound)

Here we present our framework for **Deepfake General Audio Detection**, which aims to identify whether the audio is genuine or deepfake and to locate deepfake regions. Specifically, we:

* Propose the task of deepfake general audio detection and established a benchmark for evaluation.
* Design an audio manipulation pipeline to regenerate key regions, resulting in a large quantity of convincingly realistic deepfake general audio.
* Provide a dataset, **FakeSound**, for training and evaluation for deepfake general audio detection task.
* Propose a deepfake detection model which outperforms the state-of-the-art models in previous speech deepfake competitions and human beings.


## Install dependencies
Install dependencies:
```shell
git clone https://github.com/FakeSoundData/FakeSound
conda install --yes --file requirements.txt
```

Install pre-trained models *EAT* into the *model/* directory.
```shell
cd models
mkdir EAT
cd EAT
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
git clone https://github.com/cwx-worst-one/EAT
```

## Data Preparation
Due to copyright issues, we are unable to provide the original AudioCaps audio data.
You can download the raw audio from [AudioCaps](https://audiocaps.github.io/).
The manipulated audio can be downloaded from (1) [HuggingfaceDataset](https://huggingface.co/datasets/ZeyuXie/FakeSound/tree/main) or (2) [FakeSound](https://pan.baidu.com/s/1MlwCQHfniO8jFUw7-fsFJg?pwd=fake), with the extraction code "fake".

We provide the results of the Grounding model for key region detection. 
**You can also reproduce FakeSound dataset** by regenerating key regions based on the results of the grounding,  using audio generation models [AudioLDM](https://github.com/haoheliu/AudioLDM)/[AudioLDM2](https://github.com/haoheliu/audioldm2) and super resolution model [AudioSR](https://github.com/haoheliu/versatile_audio_super_resolution).  
The metadata for the training and test sets is contained in the file *"deepfake_data/{}.json"*, where 
* the *"audio_id"* format is *{AudioCaps_id}{onset}{offset}* or *{AudioCaps_id}*, 
* the *"label"* is *"0"* for deepfake audio, with reconstructed regions indicated as *"onset_offset"*.  

## Train & Inference
The training and testing codes are named *train.py* and *inference.py*, respectively. You need to modify the *WORKSPACE_PATH* inside them to match your own directory path.
```python
  python train.py --train_file FakeSound/meta_data/train.json
  python inference.py 
```


## Acknowledgement
Our code referred to the [DKU speech deepfake detection](https://github.com/caizexin/speechbrain_PartialFake), [EAT](https://github.com/cwx-worst-one/EAT), [AudioLDM](https://github.com/haoheliu/AudioLDM), and [AudioLDM2](https://github.com/haoheliu/audioldm2). We appreciate their open-sourcing of their code.


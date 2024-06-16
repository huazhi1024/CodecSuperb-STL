# Codec-Superb-STL

This is a brief description of how to use my trained models to test the CodecSuperb hidden set.

## Content
- [Introduction](#introduction)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
- [Contact](#contact)

## Introduction
My training framework is based on [descriptinc/descript-audio-codec](https://github.com/descriptinc/descript-audio-codec?tab=readme-ov-file). For the audio provided in the openset with different sample rates (16kHz, 44.1kHz, 48kHz), I have trained the following three models:

- **16kHz Model**: `16khz_320d4q_2kbps_weights.pth`
  - enc_ratios: [2, 4, 5, 8]
  - num_quantizer: 4
  - bitrates: 2kbps

- **44.1kHz Model**: `44khz_512d8q_7kbps_weights.pth`
  - enc_ratios: [2, 4, 8, 8]
  - num_quantizer: 8
  - bitrates: 7kbps

- **48kHz Model**: `48khz_512d8q_7.5kbps_weights.pth`
  - enc_ratios: [2, 4, 8, 8]
  - num_quantizer: 8
  - bitrates: 7.5kbps

## Environment Setup

### Create Virtual Environment
```sh
conda create -n envname python=3.8
source activate envname
```

### Install Dependencies

#### CUDA and Torch versions
- CUDA: 11.6 + Cudnn 11
- Torch: `torch-1.13.0+cu116-cp38-cp38-linux_x86_64.whl`
- Torchaudio: `torchaudio-0.13.0+cu116-cp38-cp38-linux_x86_64.whl`

#### Other Dependencies
```sh
pip install -r requirements.txt
```

## Usage

1. **Clone the repository or download the project code directly**
```sh
git clone https://github.com/huazhi1024/codecsuperb-stl.git
```

2. **Format the test set as `wavname.wav /path/ref_data/wavname.wav`**
   Example command:
   ```sh
   name=$1
   find ${PWD}/${name}/ -iname "*.wav" | awk -F '/' '{print $NF, $0}' | sort > ${name}.wav.scp
   ```

3. **Navigate to the project directory and modify the test script**
   ```sh
   cd codecsuperb_src
   ```
   Modify the paths in `zhy_test_codecsuperb_all.py`:
   - Paths to the 3 models
   - Path to the scpfile

4. **Run the following command for inference to generate the reconstructed audio for the test set**
   ```sh
   python zhy_test_codecsuperb_all.py
   ```

## Contact
If you have any questions or need further assistance, please contact me at: hyzhang20@whu.edu.cn.

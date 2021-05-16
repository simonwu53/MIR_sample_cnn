# Course Project for Music Information Retrieval
Sample CNN in PyTorch implementation. Original paper: "[*Sample-level Deep
Convolutional Neural Networks for Music Auto-tagging Using Raw
Waveforms*][1]"  
The implementation is based on [*Sample CNN*][4] by Taejun.

## Table of Contents
* [Requirements / Dependencies](#requirements)
* [Dataset / Preprocessing](#preprocessing)
* [Training](#training)
* [Testing](#testing)
* [Evaluation with songs](#evaluation)
* [Results](#results)
* [Disable librosa warning messages while loading mp3 files](#librosaissue)
* [Future Works](#future)

<a name="requirements"></a>
## Requirements / Dependencies
The packages can be installed via Pip or install via `pip install -r requirements.txt`:
* Nvidia GPU only
* [PyTorch][1] (~1.8.1, **Note**: to avoid ambiguity, this package is not listed in the requirements.txt, 
  please check the official site to install the version correspond to your specific CUDA version)
* [Rich][2] (better console printing, logging, error handling)
* Pandas (reading annotations)
* librosa (audio processing)

It is recommended to do all the following steps in a `virtualenv`.

<a name="preprocessing"></a>
## Dataset / Preprocessing
* Download "The MagnaTagATune Dataset"(MTT) from [here][5]. You'll need all "Audio data" mp3 files (file1, file2, file3) 
  , a "Tag annotations" csv file, and a "Clip metadata" csv file.
  ```shell
  mp3.zip.001
  mp3.zip.002
  mp3.zip.003
  annotations_final.csv
  clip_info_final.csv
  ```
* Unzip the audio `.zip` files and merge them (reference [here][6]):
  ```shell
  cat mp3.zip.* > mp3_all.zip
  unzip mp3_all.zip
  ```
* Now you should have 16 directories in the root folder named from `0` to `f` in hexadecimal order. 
  In convention, directory `0` to `b` is used for training, directory `c` is for validation, and 
  directory `d` to `f` for testing. The directory structure should look like as below 
  (`dataset/` is root folder):
  ```
  dataset
  ├── annotations_final.csv
  ├── clip_info_final.csv
  └── raw
    ├── 0
    ├── 1
    ├── ...
    └── f
  ```
* Change the variables based on your setup in `scripts/build_mtt.sh`. An example of such variables is shown below:
  ```shell
  BASE_DIR="path/to/mtt/base"  # root directory of your dataset
  N_WORKER=12                  # number of worker processes to use
  ```
* Now run the script via `bash scripts/build_mtt.sh`, you will get the final processed dataset 
  (about 50G of size, takes about 25 mins)
  
Now your dataset directory should look like this:
```
dataset
  ├── annotations_final.csv
  ├── clip_info_final.csv
  ├── raw
    ├── 0
    ├── 1
    ├── ...
    └── f
  └── processed
    ├── train
      ├── clip-2-seg-1-of-10.npz
      ├── ...
      └── clip-58801-seg-10-of-10.npz
    ├── val
      ├── clip-*-seg-*-of-*.npz
      └── ...
    └── test
      ├── clip-*-seg-*-of-*.npz
      └── ...
```

*Optional* step:
* In order to use data normalization during training, you need to pre-calculate the training data's mean 
  and std values. You can modify the `BASE_DIR` variable and run script `bash scripts/mtt_stats.sh`. 
  The statistics results will be printed in console.

<a name="training"></a>
## Training
* Modify the arguments in `scripts/train.sh` (check args in ***main.py***)
* Run `bash scripts/train.sh` or alternatively run `bash scripts/train_nohup.sh` 
  if you wish to run the task in the background on a remote machine

<a name="testing"></a>
## Testing
* Modify the arguments in `scripts/test.sh` (check args in ***main.py***)
* Run `bash scripts/test.sh`
* The results show the AUC metric for tags (column-wise), AUC metric for samples (row-wise), 
  and averaged precision for tags and samples as well. 

<a name="evaluation"></a>
## Evaluation with songs
* Modify the arguments in `scripts/eval.sh` (check args in ***main.py***. 
  you need to provide a pre-trained model, a testing song in MTT dataset, 
  and a threshold for activating a tag as positive)
* Run `bash scripts/eval.sh`
* The results show the ground truth labels and the predicted labels with frequency

<a name="results"></a>
## Results
| Model | Configurations | Validation Loss | AUC (tags) | AUC (samples) |
| --- | --- | --- | --- | --- |
| 3^9 | Original setup | 0.14668 | 0.7191 |  0.7835 |
| 3^9 | AdamW optimizer | 0.143123 | 0.7422 | 0.7833 |
| 3^9 | AdamW + Data normalization | **0.142773** | **0.8761** | **0.9138** |
| 3^8 | AdamW + Data normalization | 0.145101 | 0.8720 | 0.9111 |
| 3^7 | AdamW + Data normalization | 0.147516 | 0.8671 | 0.9057 |

<a name="librosaissue"></a>
## Disable librosa warning messages while loading mp3 files
* find source file `audio.py` at `site-packages/librosa/core/` of your Python interpreter.
* comment line #162 which throws the warning `warnings.warn("PySoundFile failed. Trying audioread instead.")`.

<a name="future"></a>
## Future Works
* Optimized dataset preprocessing (save the processed audio in an effective way both in terms of the 
  space consumption and the loading speed)
* Other possible auto-tagging models for raw waveforms (e.g. the successor [Sample-CNN][7] model).


[1]: https://pytorch.org/
[2]: https://github.com/willmcgugan/rich
[3]: https://arxiv.org/abs/1703.01789
[4]: https://github.com/tae-jun/sample-cnn
[5]: https://mirg.city.ac.uk/codeapps/the-magnatagatune-dataset
[6]: https://github.com/keunwoochoi/magnatagatune-list
[7]: https://github.com/tae-jun/resemul

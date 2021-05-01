# Course Project for Music Information Retrieval
Replica of Sample CNN in PyTorch implementation. Original paper: "[*Sample-level Deep
Convolutional Neural Networks for Music Auto-tagging Using Raw
Waveforms*][1]"

## Table of Contents
* [Requirements / Dependencies](#requirements)

<a name="requirements"></a>
## Requirements / Dependencies
* [PyTorch][1] (~1.8.1, CUDA)
* [Rich][2] (better console printing, logging, error handling)
* Pandas (reading annotations)
* librosa (audio processing)

## Disable librosa warning messages while loading mp3 files
* find source file `audio.py` at `site-packages/librosa/core/` of your Python interpreter.
* comment line #162 which throws the warning `warnings.warn("PySoundFile failed. Trying audioread instead.")`.


[1]: https://pytorch.org/
[2]: https://github.com/willmcgugan/rich
[3]: https://arxiv.org/abs/1703.01789
# TAT

**T**ranscriptomics-to-**A**ctivity **T**ransformer is a deep learning
model to predict compound bioactivity in a dose-response assay using
compound-induced transcriptomic profiles over concentration.

## Contents

- `tat`: Python source code for TAT

## System requirements

### Hardware
#### GPUs

We have tested TAT on machines with the following GPUs:

- NVIDIA Tesla V100
- NVIDIA Ampere A100

### Software

#### Operating systems

We have tested TAT on machines with the following systems:

- Red Hat Enterprise Linux 8
- CentOS Linux 7


#### Software dependencies

- python 3.8.15
- pandas 1.5.2
- numpy 1.23.5
- pytorch 1.12.1.post200
- rdkit 2022.09.3
- scikit-learn 1.2.0
- matplotlib 3.6.2
- seaborn 0.12.2
- skorch 0.9.0


## Installation

* Install the python libraries mentioned in **Software dependencies**
  above into your python environment.

## Example dataset

An example dataset with transcriptional signatures over concentration
can be downloaded from https://broad.io/rosetta/. The example dataset
is `LINCS-Pilot1`.

## Training and validating a model

With the example LINCS dataset, we show how to build a TAT model that
takes as input the transcriptional signatures over concentration of
compounds to predict a compound-induced morphological feature in a
Cell Painting assay.

Make sure to modify the data directory path in `preprocess.py` to
ensure that the code finds the LINCS data.

```
cd ./tat
python preprocess.py
python model_build.py
```


## License

Copyright 2024 Novartis Institutes for BioMedical Research Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


See LICENSE.txt

# Contact

william_jose.godinez_navarro@novartis.com






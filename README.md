# Zebra Finch Call-Type Perception, Behavior and Acoustic

## Data

The data contains the acoustic feature data and the behavioral responses of zebra finches in auditory discrimination tasks.

### Data File Summary

```shell
data/
  behavior/
    TrialData.csv     # for each bird and date for Task 1
  confusion/
    TrialData.csv    # for each bird and each test based on stimuli in Task 2
  acoustic/
    vocParamTable.h5       # The predefined acoustic features (PAFs) obtained with BioSound (soundsig)
    vocTypeSpectroData.mat # The spectrogram features
    vocTypeSpectroBird.mat # The results of the linear discrimination analysis (LDA) on spectrographic features using a leave one bird out cross-validation procedure.  The confusion matrices for each bird contain the number of calls correctly or incorrectly classified.

 
```

## Code

This repository contains all the code necessary to perform the acoustical and behavioral analyses for the manuscript entitled "Categorical and semantic perception of the meaning of call-types in zebra finches" by Elie et al.

```shell
code/      # Python functions written for these analyses performed here
pythonNotebooks/  # Jupyter notebooks that generate all of the plots shown in the manuscript and perform most of the analyses
RNotebooks/       # R Scripts for mixed-effect statistical models.
legacyCode/       # The acoustic features and the supervised classification results were obtained in prior work and those data are in data/acoustic.  This code could be used to re-extract those features and re-perform the LDA using the raw sound files from the call data base.  The code would have to be modified so that path matches and the raw sound files are not in this repository but in the Figshare Repository (Dataset. https://doi.org/10.6084/m9.figshare.11905533.v1).  Please contact the corresponding authors for help if you want to redo those steps as well.


```


### Dependencies

Code was run and tested on Python3.11, using requirements listed in requirements.txt (can be installed with `pip install -r requirements.txt`). A local installation of R is required for running the R scripts.

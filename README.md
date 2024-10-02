<p align="center">
  <img src="header.png" width="300">
</p>

<strong><p align="center">Biochemical-free enrichment or depletion of RNA classes in real-time during direct RNA sequencing</p></strong>

<p align="center">
  <img src="arch.png">
</p>

RISER accurately classifies RNA molecules live during sequencing, directly from the first ~70-280nt-worth of raw nanopore signals with a convolutional neural network, without the need for basecalling or a reference. Depending on the user's chosen target for enrichment or depletion, RISER then either allows the molecule to complete sequencing or sends a reject command to the sequencing platform via Oxford Nanopore Technologies' [ReadUntil API](https://github.com/nanoporetech/read_until_api) to eject unwanted RNAs from the pore and conserve sequencing capacity for the RNA molecules of interest.

**[UPDATE] 2nd October 2024: RISER now supports the new RNA004 sequencing kit from ONT, in addition to the RNA002 (R9.4.1) kit. The sequencing kit can be specified with a new "--kit" parameter.**

If you use this software, please cite:
> Sneddon, A., Ravindran, A., Shanmuganandam, S. et al. Biochemical-free enrichment or depletion of RNA classes in real-time during direct RNA sequencing with RISER. Nat Commun 15, 4422 (2024). https://doi.org/10.1038/s41467-024-48673-8

# Contents

- [RNA classes supported](#rna-classes-supported)
- [Installation](#installation)
  - [Environment](#environment)
  - [Dependencies](#dependencies)
- [Test before live sequencing](#test-before-live-sequencing)
  - [Setup MinKNOW playback](#setup-minknow-playback)
  - [Test RISER with MinKNOW playback](#test-riser-with-minknow-playback)
- [Use during live sequencing](#use-during-live-sequencing)
  - [Command structure](#command-structure)
  - [Output files](#output-files)
- [Retrain for other RNA classes](#retrain-for-other-rna-classes)
  - [Preparation for retraining](#preparation-for-retraining)
  - [Retraining and testing](#retraining-and-testing)
  - [Add new model to RISER](#add-new-model-to-riser)
- [Extend RISER to new sequencing platforms](#extend-riser-to-new-sequencing-platforms)


# RNA classes supported

RISER provides models to target the following RNA classes, which are typically highly abundant in various cell lines:
* Messenger RNA (mRNA)
* Mitochondrial RNA (mtRNA)
* Globin mRNA (globin)

Users can also target their own RNA classes, by retraining the RISER model (instructions below).


# Installation

## Environment
RISER has so far been tested with the following hardware/software configuration. Usage in other configurations may be possible but should be tested using [playback mode](#setup-minknow-playback) first.

* **Operating system:** Linux (Ubuntu v18.04, v20.04, v22.04)
* **MinKNOW core:** >= 5.7.2 (the MinKNOW core version must be compatible with your MinKNOW GUI version, which usually means they need to have the same major version number, e.g., 5.* or 6.* - otherwise you will get an rpc error when attempting to run RISER). To determine MinKNOW core version on Ubuntu:
  ```
  dpkg -s minion-nc
  ```
* **Sequencing platform:** MinION Mk1B (RNA002, RNA004)
* **Python version:** 3.8.0, 3.10.8


## Dependencies

1. Set up virtual environment

   ```
   cd <path/to/riser>
   mkdir .riser-venv
   cd .riser-venv
   python3 -m venv .
   source bin/activate
   ```

2. Install RISER dependencies

   ```
   cd <path/to/riser>
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

# Test before live sequencing

**Important: Make sure you do this test first to ensure communication with the sequencing device is working before applying RISER to a live sequencing run.**

## Setup MinKNOW playback

Without wasting resources on a live sequencing run, RISER can be tested using MinKNOW's "playback" feature, which replays a bulk fast5 file recorded from a previous sequencing run to mimic data being streamed from a sequencer.
1. Obtain a bulk fast5 file (generated with either FLO-MIN106 for RNA002 or FLO-MIN004RA for RNA004) for playback in MinKNOW.
2. Open the corresponding sequencing TOML file (either *sequencing_MIN106_RNA.toml* or *sequencing_MIN004RA_RNA.toml*), found in `/opt/ont/minknow/conf/package/sequencing`.
3. In the **[custom_settings]** section, add a field:
   ```
   simulation = "/full/path/to/bulk.fast5"
   ```
4. Save the toml file.
5. Apply the above changes to enable playback by selecting "Reload Scripts" on the Start Sequencing page (top right-hand corner menu).
6. Insert a configuration test flowcell into the sequencer.
7. Start a sequencing run as usual, using the flowcell FLO-MIN106 or FLO-MIN004RA.
8. Once the run starts, a pore scan will take a few minutes. Once this is complete, you can run RISER (next section).

## Test RISER with MinKNOW playback

Now you can check that RISER can communicate with the sequencing hardware to reject molecules.
1. Make sure the steps in "Setup MinKNOW playback" are done first.
2. Run RISER. The below will selectively reject (deplete) molecules that RISER predicts to be mRNA. The script will run for 0.1 hours (this can be modified as desired with the `--duration` parameter). Please select the kit that the bulk fast5 file was generated with.
   ```
   cd <path/to/riser/riser>
   source .venv/bin/activate
   python3 riser.py --target mRNA --mode deplete --duration 0.1 --kit RNA004
   ```
4. You should see a message in MinKNOW's System Messages stating that RISER is now controlling the run.
5. For a quick check that RISER is rejecting molecules, 
6. Since a playback run simply replays the signals recorded in the bulk fast5 file, it cannot mimic reads being physically ejected from the pore. Instead, the signal is simply clipped upon receiving a reject command. Therefore, the effect of RISER can be tested by assessing the average length of reads that are mRNA and non-mRNA. The expectation is that the average length of non-mRNA reads are longer than mRNA reads, which are being rejected.
7. **Remember to remove the "simulation" field from the toml file and "Reload scripts" after testing, before sequencing your sample!**


# Use during live sequencing

1. Start a sequencing run as usual.
2. Once the initial pore scan has completed, in a terminal window run RISER (command structure detailed below).
   ```
   cd <path/to/riser/riser>
   source .venv/bin/activate
   python3 riser.py --target {target} --mode {mode} --duration {duration} --kit {kit}
   ```
4. You should see a message in the System Messages page on MinKNOW stating that RISER is now controlling the run.

## Command structure

```
usage: riser.py [-h] -t -m -d -k [-p]

optional arguments:
  -h, --help             Show this help message and exit.
  -t, --target           RNA class(es) to target for enrichment or depletion. This must be one or more of {mRNA,mtRNA,globin}. (required)
  -m, --mode             Whether to enrich or deplete the target class(es). This must be one of {enrich,deplete}. (required)
  -d, --duration         Length of time (in hours) to run RISER for. This should be
                         the same as the MinKNOW run length. (required)
  -k, --kit              Sequencing kit. This must be one of {RNA002, RNA004}. (required)
  -p, --prob_threshold   Probability threshold for classifer [0,1]. (default: 0.9)
```

**Example**

To deplete both mRNA and mtRNA in a 48h sequencing run using the RNA004 sequencing kit:
```
cd <path/to/riser/riser>
source .venv/bin/activate
python3 riser.py -t mRNA mtRNA -m deplete -d 48 -k RNA004
```

## Output files

### Console output

While running RISER, you will receive real-time progress updates:

```
Using cuda device
Usage: riser.py --target mRNA mtRNA --mode deplete --duration 24 --kit RNA002
All settings used (including those set by default):
--target        : ['mRNA', 'mtRNA']
--mode          : deplete
--duration_h    : 24
--kit           : RNA002
--threshold     : 0.9
Client is running.
Batch of 110 reads received: 59 long enough to assess, 46 of which were rejected (took 0.3148s)
Batch of  93 reads received: 29 long enough to assess, 21 of which were rejected (took 0.1376s)
Batch of 107 reads received: 32 long enough to assess, 24 of which were rejected (took 0.1568s)
...
```

### Logs

A log file named `riser_[datetime].log` will be generated in `/path/to/riser` each time you run RISER.  It will contain a more detailed version of the console output.


### CSV file with read decisions

A CSV file named `riser_[datetime].csv` will be generated in `/path/to/riser` each time you run RISER.  It will contain details of the accept/reject decision made for each read.

The possible decisions for each read are as follows:
* **Reject**: A reject request was sent to the sequencing hardware for this molecule.
* **Accept**: This molecule was sequence to completion.
* **Try again**: The RISER prediction was not confident enough (i.e., did not exceed the probability threshold argument) to make a decision for this signal, RISER will try again the next time that read is retrieved from the sequencing device, after more of the molecule has transited.
* **No decision**: The RISER prediction was not confident enough (i.e., did not exceed the probability threshold argument) to make a decision for the signal and the molecule had already transited beyond the maximum input length to RISER.

E.g., (Not all columns shown for brevity):

| read_id                              | channel | prob_targets          | threshold          | mode       | decision    |
|--------------------------------------|---------|-----------------------|--------------------|------------|-------------|
| 075391a9-2816-45b0-aebb-12b1f398fcd3 | 204     | 0.94                  | 0.9                | deplete    | reject      |
| afcbd456-1843-4322-85d2-7f001ef0dc01 | 176     | 0.83                  | 0.9                | deplete    | no_decision |
| d8de6be4-4a01-42dc-b8ab-77abc8f818e1 | 373     | 0.05                  | 0.9                | deplete    | accept      |
| 02cdeae6-3d5d-4615-bc61-7d5dd9a7217c | 91      | 0.92                  | 0.9                | deplete    | reject      |
| 4460c783-4663-4666-be4d-c52590fdff31 | 293     | 0.99                  | 0.9                | deplete    | reject      |
...

# Retrain for other RNA classes

The training code is provided to retrain RISER to target other RNA classes.

## Preparation for retraining

1. Consider whether the RNA class you would like to enrich/deplete is appropriate for RISER. For RISER to work, the RNA class needs to have unique features (e.g. sequence or biochemical modifications) in its 3' end to enable the distinction from other RNAs using the raw nanopore signal (e.g. mRNAs share common motif configurations in their 3' UTR).
2. Prepare two sets of fast5 files: 1 set containing signals that can be confidently assigned to the target class, and the other set containing signals from other RNAs (refer to publication linked above for how this was done for mRNA, mtRNA and globin models). Randomly split the signals in each set into train/test/val (e.g., with 80:10:10 split) fast5 files.
3. Preprocess all fast5 files using BoostNano to remove the sequencing adapter and poly(A) tail: <https://github.com/haotianteng/BoostNano>.
   ```
   python3 boostnano_eval.py -i $FAST5_DIR -o $OUT_DIR -m $MODEL_DIR --replace
   ```
5. Run `riser/retrain/preprocess.py` to convert the fast5 signals into numpy files. Note: The script should be run 3 times with input lengths of 2s, 3s and 4s to ensure the training data reflects the varying signal lengths streamed from the nanopore.
   ```
   source /path/to/riser/.venv/bin/activate
   SCRIPT='/path/to/riser/riser/retrain/preprocess.py'
   FAST5_DIR='/path/to/boostnano/processed/fast5s'
  
   SECS=4
   python3 $SCRIPT $SECS "$FAST5_DIR"
  
   SECS=3
   python3 $SCRIPT $SECS "$FAST5_DIR"
  
   SECS=2
   python3 $SCRIPT $SECS "$FAST5_DIR"
   ```
6. Run `riser/retrain/write_tensors.py` to convert the numpy files into PyTorch tensors for training, for each train/test/val dataset for each signal length.
   ```
   source /path/to/riser/.venv/bin/activate
   SCRIPT='/path/to/riser/riser/retrain/write_tensors.py'
  
   NPY_DIR='/path/to/numpy/files'
   python3 $SCRIPT $NPY_DIR
   ```

## Retraining and testing
1. Run `riser/train.py` to train the RISER model on the new RNA class. The yaml file allows you to configure the training parameters (e.g. number of epochs, learning rate). You can copy and modify any of the RISER yaml files (found at `/path/to/riser/riser/model/*_config_*.yaml`) - but do not modify any of the cnn parameters, as this will break the model.
   ```
   source /path/to/riser/.venv/bin/activate
   SCRIPT='/path/to/riser/riser/train.py'
   EXP_DIR='/path/to/your/experiment/directory'
   DATA_DIR='/path/to/tensors'
   CHECKPT=None
   CONFIG="/path/to/config.yaml"
   EPOCH=0
  
   python3 $SCRIPT $EXP_DIR $DATA_DIR $CHECKPT $CONFIG $EPOCH
   ```
2. Run `riser/test.py` to evaluate the performance of the newly trained model.
   ```
   source /path/to/riser/.venv/bin/activate
   SCRIPT='/path/to/riser/riser/test.py'
   MODEL_FILE='/path/to/best_model.pth' # Model trained in previous step
   CONFIG_FILE='/path/to/config.yaml' # Config file used for training in previous step
   F5_DIR='/path/to/boostnano/processed/test/fast5s'
   python3 $SCRIPT $F5_DIR $MODEL_FILE $CONFIG_FILE Y -1
   ```
3. Process the .tsv results file from step (2) to compute accuracy.

## Add new model to RISER
1. Add model.pth file to `/path/to/riser/riser/model`, with correspondingly named config.yaml file (see provided models and config files for naming convention).
2. Update the arg parser in `/path/to/riser/riser/riser.py` (main function) to include your new RNA class in the `--target` choices list.
3. You're ready to use RISER with the new model.
 
# Extend RISER to new sequencing platforms

Since different sequencing platforms produce signals with different characteristics (e.g. sampling rate, translocation speed, etc.) to adapt RISER to new sequencing platforms, the following needs to be performed:
1. Train and add a new model to RISER, trained using signals from the new hardware (as per model retraining instructions above). It may also be worth exploring other variants of the CNN hyperparameter configuration that may be better suited to the new signal type.
2. Update the poly(A) detection algorithm (specifically, the thresholds for detecting the low-variance poly(A) signal using mean and median absolute deviation will need to be checked/updated in `/path/to/riser/preprocess.py`).

The RISER software has been designed with ease of maintainability and adaptability in mind, so changes in the other classes should not be required.

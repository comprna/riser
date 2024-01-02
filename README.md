# RISER

Biochemical-free enrichment or depletion of RNA classes in real-time during direct RNA sequencing with RISER

> Preprint: <https://www.biorxiv.org/content/10.1101/2022.11.29.518281v1>

# Overview

RISER allows nanopore direct RNA sequencing runs to be targeted for the enrichment or depletion of RNA classes. RISER accurately classifies RNA molecules live during sequencing, directly from the first 2-4s of raw nanopore signals with a convolutional neural network, without the need for basecalling or a reference. Depending on the user's chosen target for enrichment or depletion, RISER then either allows the molecule to complete sequencing or sends a reject command to the sequencing platform via Oxford Nanopore Technologies' [ReadUntil API](https://github.com/nanoporetech/read_until_api) to eject unwanted RNAs from the pore and conserve sequencing capacity for the RNA molecules of interest.

Example shown: RISER enriching non-coding RNAs (such as lncRNAs) by the selective rejection of protein-coding RNAs.
![RISER architecture](architecture.png?raw=true)

# RNA classes supported

RISER provides models to target the following RNA classes, which are typically highly abundant in various cell lines:
* Messenger RNA (mRNA)
* Mitochondrial RNA (mtRNA)
* Globin mRNA (globin)

Users can also target their own RNA classes, by retraining the RISER model (instructions below).


# Installation

## Environment
RISER has so far been tested with the following sequencing environment, usage in other configurations may be possible but should be tested first.

* **Operating system:** Linux (ubuntu v18.04 and v20.04)
* **MinKNOW core:** >= 5.7.2 (the MinKNOW Core version must be compatible with your MinKNOW GUI version). To determine MinKNOW core version on Ubuntu:
  ```
  dpkg -s minion-nc
  ```
* **Sequencing platform:** MinION Mk1B (R9.4.1 flow cell)


## Dependencies

1. Set up virtual environment

   ```
   cd <path/to/riser>
   mkdir .riser-venv
   cd .riser-venv
   python3 -m venv .
   source bin/activate
   ```

2. Install PyTorch for your CUDA version: <https://pytorch.org/get-started/locally/>

3. Install RISER dependencies

   ```
   cd <path/to/riser>
   pip install --upgrade pip
   pip install -r requirements.txt
   ```


# Test before live sequencing

**Important: Make sure you do this test first to make sure everything is working properly before applying RISER to a live sequencing run.**

## Setup MinKNOW playback

Without wasting resources on a live sequencing run, RISER can be tested using MinKNOW's "playback" feature, which replays a bulk fast5 file recorded from a previous sequencing run to mimic data being streamed from a sequencer.
1. Obtain a bulk fast5 file (generated with flow cell FLO-MIN106) for playback in MinKNOW.
2. Open the sequencing TOML file *sequencing_MIN106_RNA.toml*, found in `/opt/ont/minknow/conf/package/sequencing`.
3. In the **[custom_settings]** section, add a field:
   ```
   simulation = "/full/path/to/bulk.fast5"
   ```
4. Save the toml file under a new name, e.g. *sequencing_MIN106_RNA_mod.toml*.
5. Apply the above changes to enable playback by selecting "Reload Scripts" on the Start Sequencing page (top right-hand corner menu).
6. Insert a configuration test flowcell into the sequencer.
7. Start a sequencing run as usual, using flowcell FLO-MIN106. If you have followed Step 5 correctly, then after selecting a kit you will be presented with a choice "Select the script you would like to run." Make sure you select your **_mod** file to enable playback.
8. Once the run starts, a MUX scan will take about 5 minutes.  Once this is complete, you can run RISER (next section).

## Test RISER

Now you can check that RISER is able to selectively sequence the target RNA class.
1. Make sure the steps in "Setup MinKNOW playback" are done first.
2. Run RISER. The below will selectively reject molecules that RISER predicts to be mRNA. The script will run for 1 hour (this can be modified as desired with the `--duration` parameter).
   ```
   cd <path/to/riser/riser>
   python3 riser.py --target mRNA --mode deplete --duration 1
   ```
4. You should see a message in MinKNOW's System Messages stating that RISER is now controlling the run.
5. Since a playback run simply replays the signals recorded in the bulk fast5 file, it cannot mimic reads being physically ejected from the pore. Instead, the signal is simply clipped upon receiving a reject command. Therefore, the effect of RISER can be tested by assessing the average length of reads that are mRNA and non-mRNA. The expectation is that the average length of non-mRNA reads are longer than mRNA reads, which are being rejected.


# Use RISER during live sequencing

1. Start a sequencing run as usual, using flow cell FLO-MIN106.
2. Once the initial MUX scan has completed, in a terminal window run RISER (command structure detailed below).
   ```
   cd <path/to/riser>
   source .venv/bin/activate
   python3 riser.py --target {target} --mode {mode} --duration {duration}
   ```
4. You should see a message in the System Messages page on MinKNOW stating that RISER is now controlling the run.

## Command structure

```
usage: riser.py [-h] -t -m -d [--min] [--max] [--threshold]

optional arguments:
  -h, --help       Show this help message and exit.
  -t, --target     RNA class to enrich for. This must be one or more of {mRNA,mtRNA,globin}. (required)
  -m, --mode       Whether to enrich or deplete the target class. This must be one of {enrich,deplete}. (required)
  -d, --duration   Length of time (in hours) to run RISER for. This should be
                   the same as the MinKNOW run length. (required)
  --min            Minimum number of seconds of transcript signal to use for RISER prediction. (default: 2)
  --max            Maximum number of seconds of transcript signal to try to classify before skipping this read. (default: 4)
  --threshold      Probability threshold for classifer [0,1]. (default: 0.9)
```

**Example **

To deplete mRNA in a 48h sequencing run:
```
cd <path/to/riser/riser>
source .venv/bin/activate
python3 riser.py -t mRNA -m deplete -d 48
```

## Output

### Console output

While running RISER, you will receive real-time progress updates:

```
Using cuda device
Usage: riser.py -t mRNA -m deplete -d 48
All settings used (including those set by default):
--target        : Species.NONCODING
--duration_h    : 48
--config_file   : models/cnn_best_model.yaml
--model_file    : models/cnn_best_model.pth
--polyA_length  : 6481
--secs          : 4
Client is running.
Batch of 110 reads received: 59 long enough to assess, 46 of which were rejected (took 0.3148s)
Batch of  93 reads received: 29 long enough to assess, 21 of which were rejected (took 0.1376s)
Batch of 107 reads received: 32 long enough to assess, 24 of which were rejected (took 0.1568s)
...
```

### Logs

A log file named *riser_\[datetime\].log* will be generated in `<path/to/riser>` each time you run RISER.  It will contain a more detailed version of the information sent to your console window.


### CSV file with read decisions

A CSV file named *riser_\[datetime\].csv* will be generated in `<path/to/riser>` each time you run RISER.  It will contain details of the accept/reject decision made for each read.

E.g.:

| read_id                              | channel | probability_noncoding | probability_coding | prediction | target | decision |
|--------------------------------------|---------|-----------------------|--------------------|------------|--------|----------|
| 075391a9-2816-45b0-aebb-12b1f398fcd3 | 204     | 0.83                  | 0.17               | NONCODING  | CODING | REJECT   |
| afcbd456-1843-4322-85d2-7f001ef0dc01 | 176     | 0.24                  | 0.76               | CODING     | CODING | ACCEPT   |
| d8de6be4-4a01-42dc-b8ab-77abc8f818e1 | 373     | 0.36                  | 0.64               | CODING     | CODING | ACCEPT   |
| 02cdeae6-3d5d-4615-bc61-7d5dd9a7217c | 91      | 0.79                  | 0.21               | NONCODING  | CODING | REJECT   |
| 4460c783-4663-4666-be4d-c52590fdff31 | 293     | 0.26                  | 0.74               | CODING     | CODING | ACCEPT   |
...

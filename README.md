# RISER

Real-Time In-Silico Enrichment of RNA species from Nanopore signals.

# Overview

TODO: Summary of software + diagram + paper reference

**Important: Make sure you perform the steps in Testing first to make sure everything is working properly before applying RISER to a live sequencing run.**


# Installation

## Environment

* **Operating System:** Linux

  Tested on Ubuntu v18.04 (other versions and distros need to be tested before use)
* **MinKNOW Core:** >= 4.0

  To determine MinKNOW core version on Ubuntu:
  ```
  dpkg -s minion-nc
  ```


## Dependencies

1. Set up virtual environment

   ```
   cd <path/to/riser>
   mkdir .riser-venv
   cd .riser-venv
   python3 -m venv .
   source bin/activate
   ```

2. Install dependencies

   ```
   cd <path/to/riser>
   pip install --upgrade pip
   pip install -r requirements.txt
   ```


# Testing

**Acknowledgement:** These testing instructions have been adapted from the "Testing" section of ReadFish (https://github.com/LooseLab/readfish)

## Configure MinKNOW bulk fast5 file playback

1. Obtain a bulk fast5 file (generated with flowcell FLO-MIN106) for playback in MinKNOW.

2. Open the sequencing TOML file *sequencing_MIN106_RNA.toml*, found in `/opt/ont/minknow/conf/package/sequencing`.

3. In the **[custom_settings]** section, add a field:
   ```
   simulation = "/full/path/to/bulk.fast5"
   ```

4. In the **[analysis_configuration.read_detection]** section, set the value of `break_reads_after_seconds` to 4.0

5. Save the toml file under a new name, e.g. *sequencing_MIN106_RNA_mod.toml*.  **Important: If you do not follow this step, remember to revert the changes made in steps 3 and 4 after you have finished using RISER to allow regular sequencing runs again!**

6. To apply the above changes and enable playback you will need to select "Reload Scripts" found on the Start Sequencing page (top right-hand corner menu).

7. Insert a configuration test flowcell into the sequencing device.

8. Start a sequencing run as usual, using flowcell FLO-MIN106.  If you have followed Step 5, then after selecting a kit you will be presented with a choice "Select the script you would like to run."  Make sure you select your **_mod** file to enable playback.

9. Once the run starts, a MUX scan will take about 5 minutes.  Once this is complete, observe the read length histogram.


## Test reject command

To check that RISER is able to communicate with MinKNOW and enact sequencing decisions, a simple test is to reject all reads.

1. Continue this test immediately after Step 9 of "Configure MinKNOW bulk fast5 file playback" (do not stop the sequencing run).

2. Run the reject-all script.

   ```
   cd <path/to/riser>
   python3 reject_all.py
   ```

3. Wait for a few minutes and then observe the read length histogram.  You should see a growing peak at ~200-300 b.  If you check "Split by read end reason" you should see that the peak corresponds to "Adaptive sampling voltage reversal."


## Test RNA species enrichment

Now you can check that RISER is able to selectively sequence a desired RNA species.

1. Start a new sequencing run (remember to select the **_mod** script to enable playback) and wait for the initial MUX scan to complete.

2. Run RISER.  The below will selectively sequence reads that RISER predicts to be protein-coding and will reject reads predicted to be non-coding.  The script will run for 6 hours (this can be modified as desired with the `--duration` parameter).

   ```
   cd <path/to/riser>
   python3 riser.py --target coding --duration 6
   ```

3. You should see a message in the System Messages page on MinKNOW stating that RISER is now controlling the run.

4. Since a playback run simply replays the signals recorded in the bulk fast5 file, it is not able to mimic reads being physically rejected from the nanopore.  Instead, the signal recorded for a read is simply clipped upon receiving a reject command.  Therefore, to test whether RISER is having an effect you will need to assess the average length of reads that are protein-coding and non-coding.  The expectation is that the average length of reads in the target species will be longer than those that are off target. *A script to automate this test will be provided in a future release of RISER.*


# Usage in real sequencing run

## Configure MinKNOW

1. Open the sequencing TOML file *sequencing_MIN106_RNA.toml*, found in `/opt/ont/minknow/conf/package/sequencing`.

2. In the **[analysis_configuration.read_detection]** section, set the value of `break_reads_after_seconds` to 4.0

3. Save the toml file under a new name, e.g. *sequencing_MIN106_RNA_mod.toml*.  **Important: If you do not follow this step, remember to revert the changes made in step 3 after you have finished using RISER to allow regular sequencing runs again!**

4. To apply the above changes and enable RISER you will need to select "Reload Scripts" found on the Start Sequencing page (top right-hand corner menu).

5. Start a sequencing run as usual, using flowcell FLO-MIN106.  If you have followed Step 5, then after selecting a kit you will be presented with a choice "Select the script you would like to run."  Make sure you select your **_mod** file to enable RISER.

6. Once the initial MUX scan has completed, run RISER using the commands below.

7. You should see a message in the System Messages page on MinKNOW stating that RISER is now controlling the run.

## Command structure

```
usage: riser.py [-h] -t  -d  [-c] [-m] [-p] [-s]

optional arguments:
  -h, --help        show this help message and exit
  -t , --target     RNA species to enrich for. This must be either {coding,
                    noncoding}. (required)
  -d , --duration   Length of time (in hours) to run RISER for. This should be
                    the same as the MinKNOW run length. (required)
  -c , --config     Config file for model hyperparameters. (default:
                    models/cnn_best_model.yaml)
  -m , --model      File containing saved model weights. (default:
                    models/cnn_best_model.pth)
  -p , --polya      Number of values to remove from the start of the raw
                    signal to exclude the polyA tail and sequencing adapter
                    signal from analysis. (default: 6481)
  -s , --secs       Number of seconds of transcript signal to use for
                    decision. (default: 4)
```


## Example usage

To enrich for non-coding RNA, RISER can simply be run with the following command (make sure to set the duration `-d` equal to your MinKNOW run length in hours).

```
cd <path/to/riser>
python3 riser.py -t noncoding -d 48
```

## Output

### Console output

While running RISER, you will receive real-time progress updates:

```
Using cuda device
Usage: riser.py -t noncoding -d 48
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
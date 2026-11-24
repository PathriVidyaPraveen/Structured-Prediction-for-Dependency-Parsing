## How to run

Place train.csv, test.csv and sample.csv in the same folder as main.py, then run:

    python main.py --data . --out submission.csv

This writes submission.csv in the Kaggle format (columns ID, Label), with one row
per ID in sample.csv.

### Requirements
Python 3 with numpy, pandas, scipy, scikit-learn and joblib:

    pip install numpy pandas scipy scikit-learn joblib

### Faster run on a multi-core machine
By default the script uses up to 4 CPU cores. To use more (this changes the speed
only, not the results), for example on 22 cores:

    python main.py --data . --out submission.csv --jobs 22 --tag_jobs 6 --beam_jobs 11

- --jobs: parallel jobs for the parsers
- --tag_jobs: parallel tagger fits; each needs about 2-3 GB of RAM
- --beam_jobs: parallel beam-parser trainings; each needs about 1 GB of RAM

### Runtime and reproducibility
A full run from scratch took about 7 hours on a 22-core laptop (the last command
above); with the default 4 jobs it will take considerably longer. All randomness uses
fixed seeds, and a fresh run reproduced the Kaggle submission exactly (all 227,783
rows identical).

Intermediate results are saved in a ./cache folder. If a run is interrupted,
rerunning the same command resumes from the last finished stage. Delete ./cache to
start completely from scratch.
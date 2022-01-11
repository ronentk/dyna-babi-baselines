# dyna-babi-baselines
Experiments running models on dyba-bAbI tasks

## Data
To create a ./data directory and download the bAbI dataset to it
```
./get_data.sh
```

## EntNet

### Setup

- Set up a new environment. We tested on Conda in python 3.7.
- From repo root, run `pip install -r requirements_ent.txt`

### Usage

To jointly train EntNet on a subset of bAbI tasks: 

```
python EntNet.py [--verbose] --train_tasks [int+] \
    --test_tasks [int+] [--train] [--test] \
    --try_n [int] --n_tries [int] --max_stuck_epochs=[int] \
    --optimizer_name [sgd|adam] [--no_tie_keys] --n_memories [int] \
    [--use_wandb_data] --data_dir [path\to\data\dir]
```
Saves model and optimizer to `./trained_models/`

Example:
```
python EntNet.py --verbose --train_tasks 2 11 \
    --test_tasks 2 11 --train --test \
    --try_n 0 --n_tries 1 --max_stuck_epochs=50 \
    --optimizer_name=sgd --no_tie_keys --n_memories=20 \
    --use_wandb_data --data_dir=eco-semantics/data/tasks_1-20_v1-2/en-valid-10k/
```

Note that the arguments `train_tasks` and `test_tasks` allow testing on different tasks then the training tasks.

Use `--test` to run a trained model on the test set.
If `--train` is also used, testing will be done at the end of training. 

To allow early stopping the model's loss over the dev set
doesn't improve for n epochs, use `max_stuck_epochs=n`.

The standard number of memory cells in EntNet is 20.
To use a different number n, use `n_memories=n`. 

EntNet design allows tying the memory cells "keys" to words from the vocabulary,
but this typically leads to worse results. To tie the keys, omit the option `--no_tie_keys`.

EntNet results are originally presented as best out of 10 runs.
the arguments `try_n` and `n_tries` represent the index of current
run and the number of runs respectively, and are used for documentation.

If using the `--use_wandb_data` flag, the `data_dir` should be a valid wandb dataset path

## Running SAM

### Setup

- Set up a new environment. We tested on Conda in python 3.7.
- From repo root, run `pip install -r requirements_sam.txt`

### Usage

To jointly train SAM on a subset of bAbI tasks:

```
python run_all_babi.py [--use_wandb_data] \
        --data_dir [path\to\data\dir] \
        --train_tasks [int+] --test_tasks [int+] \
        --try_n [int] --batch_size=[int]
```
Saves model to `./saved_models/`

```
python run_all_babi.py --use_wandb_data \
        --data_dir=eco-semantics/data/tasks_1-20_v1-2/en-valid-10k/ \
        --train_tasks 30 --test_tasks 30 \
        --try_n 1 --batch_size=32
```

Note that the arguments `train_tasks` and `test_tasks` allow testing on different tasks then the training tasks.

Use `--test` to run a trained model on the test set.
If `--train` is also used, testing will be done at the end of training.

SAM results are originally presented as best out of 10 runs.
the argument `try_n` represents the index of current run, and is used for documentation.

If using the `--use_wandb_data` flag, the `data_dir` should be a valid wandb dataset path

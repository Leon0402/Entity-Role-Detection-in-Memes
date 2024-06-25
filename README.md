# Entity-Role Detection in Memes

The [CheckThat! lab of CLEF 2024 conference](https://checkthat.gitlab.io/clef2024/) aims to support and understand the journalistic verification process. This repository tackles the fourth task: Entity-Role Detection in Memes. Given a meme and a list of entities, the task is to predict the role of each entity. Possible roles are “hero”, “villain”, “victim”, or “other”.

## Installation

Following dependencies are required for the project: 

* Python ^3.10 ([PyEnv](https://github.com/pyenv/pyenv) can be used)
* [Poetry](https://python-poetry.org/) ^1.8

Use Poetry to create a virtual environment and install all dependencies with:

```bash
poetry install
```

Note: For IDE support, it needs to find the virtual environment created by poetry. This works better, most of the time automatically, if you [configure poetry](https://python-poetry.org/docs/configuration/#virtualenvsin-project) to create the virtual environment within the project prior to running above command.

The virtual environment needs to be activated. One possibility is to spawn a new shell with: 

```bash
poetry shell
```

In the rest of the README it is assumed that all commands are executed within this shell.

## Data Preparation

Data can be downloaded directly with:

```bash
python -m meme_entity_detection.scripts.download_data --download-url "https://drive.usercontent.google.com/download?id=1Dqma78ofOAlVnaCehfjToJXFsFfiQUbX&export=download&confirm=t" --output-path "./data"
```

## Baseline 

Train with:
```bash
python -m meme_entity_detection.scripts.baseline fit --seed_everything 4 --data.batch_size 8 --data.data_dir "./data/HVVMemesWithFaces" --model.lr 0.00001 --trainer.max_epochs 12 --trainer.accumulate_grad_batches 2 --trainer.precision "bf16" --trainer.logger TensorBoardLogger --trainer.logger.save_dir ./logs/baseline_faces/training  --config configs/config.yaml 
```

Check training results with:
```bash
tensorboard --logdir ./logs/baseline/training
```

Test with:
```bash
python -m meme_entity_detection.scripts.baseline test --seed_everything 4 --data.batch_size 8 --data.data_dir "./data/HVVMemes" --model.lr 0.00001 --trainer.max_epochs 12 --trainer.accumulate_grad_batches 2 --trainer.precision "bf16" --trainer.logger TensorBoardLogger --trainer.logger.save_dir ./logs/baseline/test  --config configs/config.yaml --ckpt_path ./logs/baseline/training/lightning_logs/version_<version>/checkpoints/best-checkpoint.ckpt
```

Check test results with:
```bash
tensorboard --logdir ./logs/baseline/test
```

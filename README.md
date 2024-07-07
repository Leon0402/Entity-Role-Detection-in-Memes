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
python -m meme_entity_detection.scripts.download_data --download-url "https://drive.usercontent.google.com/download?id=1Dqma78ofOAlVnaCehfjToJXFsFfiQUbX&export=download&confirm=t" --output-path "."
```

This is the original data enhanced with various information such as OCR from ChatGPt-4o and descriptions from ChatGPt-4o and KOSMOS-2. The notebooks for preprocessing can be found in `meme_entity_detection.notebooks`.

## Models 

For our architecture we have a single baseline model with a flexible backbone network. Available backbones are RoBERTA, DeBERTa and ViLT. More can be easily added by implementing the model and tokenizer interface in `meme_entity_detection.model`. Similary, we have a single dataset that can be highly customized, e.g. selection of ocr used or customizing the tokenizer. Below are a few examples for the different backbones in our standard configuration with ChatGPT-4o OCR and KOSMOS-2 image descriptions. For a full documentation use command with the `--help` flag to see customizations.

## RoBERTa

Train with:
```bash
python -m meme_entity_detection.scripts.baseline fit --seed_everything 4 --data.batch_size 32 --data.data_dir "./data/HVVMemes" --data.ocr_type GPT --data.description_type KOSMOS --data.tokenizer meme_entity_detection.model.RobertaTokenizer --model.lr 0.00001 --model.backbone meme_entity_detection.model.RobertaModel --trainer.max_epochs 12 --trainer.accumulate_grad_batches 1 --trainer.precision "bf16-mixed" --trainer.logger TensorBoardLogger --trainer.logger.save_dir ./logs/roberta_faces/training  --config configs/config.yaml 
```

Check training results with:
```bash
tensorboard --logdir ./logs/roberta_faces/training
```

Test with:
```bash
python -m meme_entity_detection.scripts.baseline test --seed_everything 4 --data.batch_size 32 --data.data_dir "./data/HVVMemes" --data.ocr_type GPT --data.description_type KOSMOS --data.tokenizer meme_entity_detection.model.RobertaTokenizer --model.lr 0.00001 --model.backbone meme_entity_detection.model.RobertaModel --trainer.max_epochs 12 --trainer.accumulate_grad_batches 1 --trainer.precision "bf16-mixed" --trainer.logger TensorBoardLogger --trainer.logger.save_dir ./logs/roberta_faces/test  --config configs/config.yaml --ckpt_path ./logs/roberta_faces/training/lightning_logs/version_<version>/checkpoints/best-checkpoint.ckpt
```

Check test results with:
```bash
tensorboard --logdir ./logs/roberta_faces/test
```

## DeBERTa

Train with:
```bash
python -m meme_entity_detection.scripts.baseline fit --seed_everything 4 --data.batch_size 16 --data.data_dir "./data/HVVMemes" --data.ocr_type GPT --data.description_type KOSMOS --data.tokenizer meme_entity_detection.model.DebertaTokenizer --model.lr 0.00001 --model.backbone meme_entity_detection.model.DebertaModel --trainer.max_epochs 12 --trainer.accumulate_grad_batches 2 --trainer.precision "bf16-mixed" --trainer.logger TensorBoardLogger --trainer.logger.save_dir ./logs/deberta_faces/training  --config configs/config.yaml 
```

Check training results with:
```bash
tensorboard --logdir ./logs/deberta_faces/training
```

Test with:
```bash
python -m meme_entity_detection.scripts.baseline test --seed_everything 4 --data.batch_size 16 --data.data_dir "./data/HVVMemes" --data.ocr_type GPT --data.description_type KOSMOS --data.tokenizer meme_entity_detection.model.DebertaTokenizer --model.lr 0.00001 --model.backbone meme_entity_detection.model.DebertaModel --trainer.max_epochs 12 --trainer.accumulate_grad_batches 2 --trainer.precision "bf16-mixed" --trainer.logger TensorBoardLogger --trainer.logger.save_dir ./logs/deberta_faces/test  --config configs/config.yaml --ckpt_path ./logs/deberta_faces/training/lightning_logs/version_<version>/checkpoints/best-checkpoint.ckpt
```

Check test results with:
```bash
tensorboard --logdir ./logs/deberta_faces/test
```

## VilT

Train with:
```bash
python -m meme_entity_detection.scripts.baseline fit --seed_everything 4 --data.batch_size 32 --data.data_dir "./data/HVVMemes" --data.ocr_type GPT --data.description_type KOSMOS --data.tokenizer meme_entity_detection.model.ViltTokenizer --model.lr 0.00001 --model.backbone meme_entity_detection.model.ViltModel --trainer.max_epochs 12 --trainer.accumulate_grad_batches 1 --trainer.precision "bf16-mixed" --trainer.logger TensorBoardLogger --trainer.logger.save_dir ./logs/filt_faces/training  --config configs/config.yaml 
```

Check training results with:
```bash
tensorboard --logdir ./logs/vilt_faces/training
```

Test with:
```bash
python -m meme_entity_detection.scripts.baseline test --seed_everything 4 --data.batch_size 32 --data.data_dir "./data/HVVMemes" --data.ocr_type GPT --data.description_type KOSMOS --data.tokenizer meme_entity_detection.model.ViltTokenizer --model.lr 0.00001 --model.backbone meme_entity_detection.model.ViltModel --trainer.max_epochs 12 --trainer.accumulate_grad_batches 1 --trainer.precision "bf16-mixed" --trainer.logger TensorBoardLogger --trainer.logger.save_dir ./logs/filt_faces/test  --config configs/config.yaml --ckpt_path ./logs/vilt_faces/training/lightning_logs/version_<version>/checkpoints/best-checkpoint.ckpt
```

Check test results with:
```bash
tensorboard --logdir ./logs/vilt_faces/test
```

## GPT-4o

Classifications were generated and written into the dataset. The test step will evaluate these: 

```bash
python -m meme_entity_detection.scripts.baseline test --seed_everything 4 --data.batch_size 32 --data.data_dir "./data/HVVMemes" --data.ocr_type GPT --data.description_type KOSMOS --model.lr 0.00001 --model.backbone meme_entity_detection.model.GPT4oPromptAnswers --trainer.max_epochs 12 --trainer.accumulate_grad_batches 1 --trainer.precision "bf16-mixed" --trainer.logger TensorBoardLogger --trainer.logger.save_dir ./logs/gpt/test  --config configs/config.yaml
```

Check test results with:
```bash
tensorboard --logdir ./logs/gpt/test
```
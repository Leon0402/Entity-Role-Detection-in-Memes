import logging

import lightning.pytorch.cli
import torch

import meme_entity_detection.dataset
import meme_entity_detection.system


class MainCli(lightning.pytorch.cli.LightningCLI):

    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            '--loglevel',
            default='info',
            help='Provide logging level. Example --loglevel debug, default=info'
        )
        # TODO: Argument seed_everything can also be a boolean, in which case this code fails.
        parser.link_arguments("seed_everything", "data.seed")

    def before_fit(self):
        # self.model = torch.compile(self.model)

        loglevel = self.config['fit']['loglevel'].upper()
        logging.basicConfig(level=getattr(logging, loglevel))


def cli_main():
    torch.set_float32_matmul_precision("medium")

    cli = MainCli(
        model_class=meme_entity_detection.system.BaselineLightningModel,
        datamodule_class=meme_entity_detection.dataset.DataModule,
    )


if __name__ == "__main__":
    cli_main()

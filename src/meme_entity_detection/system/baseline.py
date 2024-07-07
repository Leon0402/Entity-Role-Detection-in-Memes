import torch
import lightning as L
import transformers
import torchmetrics

import meme_entity_detection.model
import meme_entity_detection.utils.task_properties
import meme_entity_detection.model.interface


class BaselineLightningModel(L.LightningModule):

    def __init__(
        self, lr: float = 1e-3,
        backbone: meme_entity_detection.model.interface.Model = meme_entity_detection.model.ViltModel()
    ):
        super().__init__()
        self.lr = lr
        self.model = backbone

        self.save_hyperparameters(ignore=['backbone'])

        # TODO
        # self.example_input_array = ()

        num_classes = meme_entity_detection.utils.task_properties.num_classes
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", average='macro', num_classes=num_classes)
        self.train_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=num_classes)
        self.train_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=num_classes)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=num_classes)

        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", average='macro', num_classes=num_classes)
        self.val_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=num_classes)
        self.val_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=num_classes)
        self.val_f1 = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=num_classes)

        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", average='macro', num_classes=num_classes)
        self.test_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=num_classes)
        self.test_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=num_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes)

    def forward(self, images, part_point_clouds, part_equivalence_counts):
        return self.model(images, part_point_clouds, part_equivalence_counts)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [{
            'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 2e-5
        }, {
            'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]

        optimizer = torch.optim.AdamW(optimizer_parameters, lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": transformers.get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=0, num_training_steps=self.trainer.estimated_stepping_batches
                ),
            },
        }

    def _write_log(self, mode, accuracy, precision, recall, f1):
        self.log(f'{mode}/accuracy', accuracy, on_step=False, on_epoch=True)
        self.log(f'{mode}/precision', precision, on_step=False, on_epoch=True)
        self.log(f'{mode}/recall', recall, on_step=False, on_epoch=True)
        self.log(f'{mode}/f1', f1, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        loss, preds = self.model(batch)

        self.log('train/loss', loss)

        self.train_accuracy(preds, batch["labels"])
        self.train_precision(preds, batch["labels"])
        self.train_recall(preds, batch["labels"])
        self.train_f1(preds, batch["labels"])
        self._write_log("train", self.train_accuracy, self.train_precision, self.train_recall, self.train_f1)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds = self.model(batch)

        self.log('validation/loss', loss)
        self.val_accuracy(preds, batch["labels"])
        self.val_precision(preds, batch["labels"])
        self.val_recall(preds, batch["labels"])
        self.val_f1(preds, batch["labels"])

        self._write_log("validation", self.val_accuracy, self.val_precision, self.val_recall, self.val_f1)

    def test_step(self, batch, batch_idx):
        loss, preds = self.model(batch)

        self.test_accuracy(preds, batch["labels"])
        self.test_precision(preds, batch["labels"])
        self.test_recall(preds, batch["labels"])
        self.test_f1(preds, batch["labels"])
        self.test_confusion_matrix(preds, batch["labels"])

        self._write_log("test", self.test_accuracy, self.test_precision, self.test_recall, self.test_f1)

    def on_test_epoch_end(self):

        self.logger.experiment.add_figure(
            'test/confusion_matrix',
            self.test_confusion_matrix.plot(labels=meme_entity_detection.utils.task_properties.labels)[0],
            global_step=self.global_step
        )

    # def on_before_optimizer_step(self, optimizer):
    #     self.log_dict(lightning.pytorch.utilities.grad_norm(self, norm_type=2))

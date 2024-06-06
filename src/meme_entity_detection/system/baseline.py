import torch
import lightning as L
import transformers
import torchmetrics


class BaselineLightningModel(L.LightningModule):

    def __init__(self, lr: float = 1e-3, model_name="microsoft/deberta-v3-large"):
        super().__init__()

        label2id = {'hero': 3, 'villain': 2, 'victim': 1, 'other': 0}
        num_classes = len(label2id)

        self.labels = [item[0] for item in sorted(label2id.items(), key=lambda item: item[1])]

        self.lr = lr
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.model.train()

        self.save_hyperparameters()

        # TODO
        # self.example_input_array = ()

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

        optimizer = transformers.AdamW(optimizer_parameters, lr=self.lr)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": transformers.get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=0, num_training_steps=self.trainer.estimated_stepping_batches
                ),
            },
        }

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        sentiment_target = batch['labels']

        output = self.model(
            input_ids,
            attention_mask,
            token_type_ids,
            labels=sentiment_target,
        )

        self.log('train/loss', output.loss, batch_size=len(input_ids))

        preds = torch.argmax(output.logits, dim=1)
        self.train_accuracy(preds, sentiment_target)
        self.train_precision(preds, sentiment_target)
        self.train_recall(preds, sentiment_target)
        self.train_f1(preds, sentiment_target)

        self.log('train/accuracy', self.train_accuracy, on_step=False, on_epoch=True)
        self.log('train/precision', self.train_precision, on_step=False, on_epoch=True)
        self.log('train/recall', self.train_recall, on_step=False, on_epoch=True)
        self.log('train/f1', self.train_f1, on_step=False, on_epoch=True)

        return output.loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        sentiment_target = batch['labels']

        output = self.model(
            input_ids,
            attention_mask,
            token_type_ids,
            labels=sentiment_target,
        )

        self.log('validation/loss', output.loss, batch_size=len(input_ids))

        preds = torch.argmax(output.logits, dim=1)
        self.val_accuracy(preds, sentiment_target)
        self.val_precision(preds, sentiment_target)
        self.val_recall(preds, sentiment_target)
        self.val_f1(preds, sentiment_target)

        self.log('validation/accuracy', self.val_accuracy, on_step=False, on_epoch=True)
        self.log('validation/precision', self.val_precision, on_step=False, on_epoch=True)
        self.log('validation/recall', self.val_recall, on_step=False, on_epoch=True)
        self.log('validation/f1', self.val_f1, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        sentiment_target = batch['labels']

        output = self.model(
            input_ids,
            attention_mask,
            token_type_ids,
            labels=sentiment_target,
        )

        preds = torch.argmax(output.logits, dim=1)
        self.test_accuracy(preds, sentiment_target)
        self.test_precision(preds, sentiment_target)
        self.test_recall(preds, sentiment_target)
        self.test_f1(preds, sentiment_target)
        self.test_confusion_matrix(preds, sentiment_target)

        self.log('test/accuracy', self.test_accuracy, on_step=False, on_epoch=True)
        self.log('test/precision', self.test_precision, on_step=False, on_epoch=True)
        self.log('test/recall', self.test_recall, on_step=False, on_epoch=True)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):

        self.logger.experiment.add_figure(
            'test/confusion_matrix',
            self.test_confusion_matrix.plot(labels=self.labels)[0], global_step=self.global_step
        )

    # def on_before_optimizer_step(self, optimizer):
    #     self.log_dict(lightning.pytorch.utilities.grad_norm(self, norm_type=2))

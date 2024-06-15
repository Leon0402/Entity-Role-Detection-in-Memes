import torch
import lightning as L
import transformers
import torchmetrics


class BaselineLightningModel(L.LightningModule):

    def __init__(self, lr: float = 1e-3, model_name: str ="microsoft/deberta-v3-large"):
        super().__init__()

        label2id = {'hero': 3, 'villain': 2, 'victim': 1, 'other': 0}
        num_classes = len(label2id)

        self.labels = [item[0] for item in sorted(label2id.items(), key=lambda item: item[1])]
        
        if "deberta" in model_name.lower():
            self.model_type = "deberta"
        elif "roberta" in model_name.lower():
            self.model_type = "roberta"

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

    def _output_dependend_on_model_type(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        sentiment_target = batch['labels']
        
        if self.model_type == "deberta":
            token_type_ids = batch['token_type_ids']
            output = self.model(            
                input_ids,
                attention_mask,
                token_type_ids,
                labels=sentiment_target,
            ) 
        
        elif self.model_type == "roberta":
            output = self.model(
                input_ids, 
                attention_mask, 
                labels=sentiment_target
            )
           
        return output, input_ids, sentiment_target
    
    def _write_log(self, mode, accuracy, precision, recall, f1):
        self.log(f'{mode}/accuracy', accuracy, on_step=False, on_epoch=True)
        self.log(f'{mode}/precision', precision, on_step=False, on_epoch=True)
        self.log(f'{mode}/recall', recall, on_step=False, on_epoch=True)
        self.log(f'{mode}/f1', f1, on_step=False, on_epoch=True)
        
    def training_step(self, batch, batch_idx):
        output, input_ids, sentiment_target = self._output_dependend_on_model_type(batch)
            
        self.log('train/loss', output.loss, batch_size=len(input_ids))

        preds = torch.argmax(output.logits, dim=1)
        self.train_accuracy(preds, sentiment_target)
        self.train_precision(preds, sentiment_target)
        self.train_recall(preds, sentiment_target)
        self.train_f1(preds, sentiment_target)

        self._write_log("train", self.train_accuracy, self.train_precision, self.train_recall, self.train_f1)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output, input_ids, sentiment_target = self._output_dependend_on_model_type(batch)

        self.log('validation/loss', output.loss, batch_size=len(input_ids))

        preds = torch.argmax(output.logits, dim=1)
        self.val_accuracy(preds, sentiment_target)
        self.val_precision(preds, sentiment_target)
        self.val_recall(preds, sentiment_target)
        self.val_f1(preds, sentiment_target)

        self._write_log("validation", self.val_accuracy, self.val_precision, self.val_recall, self.val_f1)

    def test_step(self, batch, batch_idx):
        output, input_ids, sentiment_target = self._output_dependend_on_model_type(batch)

        preds = torch.argmax(output.logits, dim=1)
        self.test_accuracy(preds, sentiment_target)
        self.test_precision(preds, sentiment_target)
        self.test_recall(preds, sentiment_target)
        self.test_f1(preds, sentiment_target)
        self.test_confusion_matrix(preds, sentiment_target)

        self._write_log("test", self.test_accuracy, self.test_precision, self.test_recall, self.test_f1)


    def on_test_epoch_end(self):

        self.logger.experiment.add_figure(
            'test/confusion_matrix',
            self.test_confusion_matrix.plot(labels=self.labels)[0], global_step=self.global_step
        )

    # def on_before_optimizer_step(self, optimizer):
    #     self.log_dict(lightning.pytorch.utilities.grad_norm(self, norm_type=2))

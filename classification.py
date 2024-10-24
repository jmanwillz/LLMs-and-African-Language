import argparse
from datetime import datetime

import evaluate
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
)

import wandb


class ClassificationTrainer:
    def __init__(
        self,
        model_checkpoint,
        dataset_name,
        tokenizer,
        is_finetuned,
        model_name,
        current_time,
        run_name,
        output_dir="./results",
        log_dir="./logs",
    ):
        self.model_checkpoint = model_checkpoint
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.is_finetuned = is_finetuned
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.model_name = model_name
        self.current_time = current_time
        self.run_name = run_name

        self.accuracy_metric = evaluate.load("accuracy")
        self.precision_metric = evaluate.load("precision")
        self.recall_metric = evaluate.load("recall")
        self.f1_metric = evaluate.load("f1")

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def prepare_dataset(self):
        dataset = load_dataset(self.dataset_name, "swa")
        tokenized_dataset = dataset.map(
            self.tokenize_function, batched=True, remove_columns=["text"]
        )
        return tokenized_dataset

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)

        accuracy = self.accuracy_metric.compute(
            predictions=predictions, references=labels
        )
        precision = self.precision_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
        recall = self.recall_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )
        f1 = self.f1_metric.compute(
            predictions=predictions, references=labels, average="weighted"
        )

        return {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
            "f1": f1["f1"],
        }

    def train(self):
        processed_dataset = self.prepare_dataset()

        if self.is_finetuned:
            print("Loading the fine-tuned model...")
            model_cls = XLMRobertaForSequenceClassification.from_pretrained(
                self.model_checkpoint, num_labels=7
            )
        else:
            print("Loading the base model...")
            model_cls = XLMRobertaForSequenceClassification.from_pretrained(
                self.model_name, num_labels=7
            )

        training_args = TrainingArguments(
            run_name=self.run_name,
            output_dir=self.output_dir,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=4,
            logging_strategy="steps",
            logging_steps=10,
            num_train_epochs=5,
            logging_dir=self.log_dir,
            report_to="wandb",
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model_cls,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
            compute_metrics=self.compute_metrics,
        )

        eval_results = trainer.evaluate(processed_dataset["test"])
        wandb.log(
            {
                "accuracy": eval_results["eval_accuracy"],
                "precision": eval_results["eval_precision"],
                "recall": eval_results["eval_recall"],
                "f1": eval_results["eval_f1"],
            }
        )

        trainer.train()

        eval_results = trainer.evaluate(processed_dataset["test"])
        wandb.log(
            {
                "accuracy": eval_results["eval_accuracy"],
                "precision": eval_results["eval_precision"],
                "recall": eval_results["eval_recall"],
                "f1": eval_results["eval_f1"],
            }
        )

        trainer.save_model(
            f"{self.output_dir}/{self.current_time}_classification_finetuned_{str(self.is_finetuned).lower()}"
        )
        trainer.save_state()


def main():
    checkpoint_folder = "./results/2024-10-19_13-49-04_mlm_finetuned_model"

    project_name = "LLMs and African Language"
    model_name = "xlm-roberta-base"
    dataset_name = "masakhane/masakhanews"

    parser = argparse.ArgumentParser(description=project_name)
    parser.add_argument(
        "--use-base",
        action="store_true",
        help="Use the base model if set, otherwise use the finetuned model.",
    )
    args = parser.parse_args()
    use_finetuned = not args.use_base

    print(f"Using the fine-tuned model: {use_finetuned}")
    if use_finetuned:
        print(f"Model located at: {checkpoint_folder}")

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{project_name.lower().replace(' ', '_')}_classification_finetuned_{str(use_finetuned).lower()}_{current_time}"

    run = wandb.init(project=project_name, name=run_name)
    run.alert(title="Run Started", text=f"The run {run_name} has started.")

    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

    classification_trainer = ClassificationTrainer(
        model_checkpoint=checkpoint_folder,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        is_finetuned=use_finetuned,
        model_name=model_name,
        current_time=current_time,
        run_name=run_name,
    )
    classification_trainer.train()

    run.finish()


if __name__ == "__main__":
    main()

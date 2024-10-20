import math
import warnings
from datetime import datetime

from datasets import load_dataset
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    XLMRobertaForMaskedLM,
    XLMRobertaTokenizer,
)

import wandb

# Suppress all warnings
warnings.filterwarnings("ignore")


class MaskedLanguageModelingTrainer:
    def __init__(
        self,
        model_checkpoint,
        dataset_name,
        tokenizer,
        current_time,
        block_size=128,
        output_dir="./results",
        log_dir="./logs",
    ):
        self.model_checkpoint = model_checkpoint
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.log_dir = log_dir
        self.block_size = block_size
        self.current_time = current_time

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def compute_perplexity(self, loss):
        return math.exp(loss) if loss < float("inf") else float("inf")

    def group_texts(self, examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // self.block_size) * self.block_size
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    def prepare_dataset(self):
        dataset = load_dataset(self.dataset_name, trust_remote_code=True)
        tokenized_dataset = dataset.map(
            self.tokenize_function, batched=True, num_proc=4, remove_columns=["text"]
        )
        lm_datasets = tokenized_dataset.map(
            self.group_texts, batched=True, batch_size=1000, num_proc=4
        )
        return lm_datasets

    def train(self):
        processed_dataset = self.prepare_dataset()
        model = XLMRobertaForMaskedLM.from_pretrained(self.model_checkpoint)
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=0.15
        )

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            logging_steps=10,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=3,
            eval_steps=50,
            save_steps=100,
            logging_dir=self.log_dir,
            report_to="wandb",
            load_best_model_at_end=True,
        )

        # small_train_dataset = (
        #     processed_dataset["train"].shuffle(seed=42).select(range(1000))
        # )
        # small_validation_dataset = (
        #     processed_dataset["validation"].shuffle(seed=42).select(range(1000))
        # )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
            data_collator=data_collator,
        )

        eval_results = trainer.evaluate()
        loss = eval_results["eval_loss"]
        perplexity = self.compute_perplexity(loss)

        # Log intial metrics to Weights and Biases
        wandb.log({"eval_loss": loss, "eval_perplexity": perplexity})

        trainer.train()

        eval_results = trainer.evaluate()
        loss = eval_results["eval_loss"]
        perplexity = self.compute_perplexity(loss)

        # Log fine tuned metrics to Weights and Biases
        wandb.log({"eval_loss": loss, "eval_perplexity": perplexity})

        trainer.save_model(f"{self.output_dir}/{self.current_time}_mlm_finetuned_model")


def main():
    project_name = "LLMs and African Language"
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{project_name.lower().replace(' ', '_')}_masked_language_modelling_{current_time}"
    model_name = "xlm-roberta-base"
    dataset_name = "uestc-swahili/swahili"

    wandb.init(project=project_name, name=run_name)

    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

    mlm_trainer = MaskedLanguageModelingTrainer(
        model_checkpoint=model_name,
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        current_time=current_time,
    )
    mlm_trainer.train()

    wandb.finish()


if __name__ == "__main__":
    main()

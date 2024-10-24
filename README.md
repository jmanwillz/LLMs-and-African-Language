# LLMs and African Language

For this project you will be aiming to assist in performing systematic reviews. A systematic review is essentially a very structured literature review and our models will have two components:

1. a binary recommendation,
2. the extraction of relevant text.

The binary recommendation is a label given to a document indicating whether the model believes it is a relevant piece of literature. A label of 1 indicates that it is relevant and 0 indicates that it is not. Secondly, pieces of relevant documents should then be highlighted if it contains important information. For your project, you must perform the systematic review in an African language other than English. To achieve this you must perform two full fine-tuning trainings of your model. The first is a more general fine-tuning of a pretrained model and can be anything you think will be helpful for the final performance of the model. The second fine-tuning is when you then train the model to identify your topic in the chosen language. You can also choose whichever topic you like - systematic reviews are generally done in medicine, however you will already be in a low-resource domain (which medicine is) by using an African language. So being low resource twice over is not a mandatory challenge.

## Specification

Please find the specification [here](./resources/Project.pdf).

## Report

Please find the report [here](./report/LLMs_and_African_Language.pdf).

## Running the Code

Step 1: Create a [conda](https://docs.conda.io/projects/conda/en/latest/index.html#) environment with the required packages.

```bash
conda env create -f environment.yml
```

Step 2: Activate the conda environment.

```bash
conda activate development
```

Step 3: Login to [Weights & Biases](https://wandb.ai/site/).

```bash
wandb login --relogin
```

Step 4: Run the initial fine-tuning of [FacebookAI/xlm-roberta-base](https://huggingface.co/FacebookAI/xlm-roberta-base) on the MLM task.

```bash
python fine_tune.py
```

Step 5: Run the second fine-tunings of the model on the classification task.

```bash
# Trains on the xlm-roberta-base model.
python classification.py --use-base
```

```bash
# Trains on the fine-tuned model from step 4.
python classification.py
```

Before running the `classification.py` file, ensure that you edit the `checkpoint_folder` variable to point to where the fine-tuned model from step 4 was saved.

## Team

| ![Jason Wille](images/jason.jpeg "Jason Wille") <br/> [Jason Wille](https://www.linkedin.com/in/jasonwille97/) | ![Reece Lazarus](images/reece.jpeg "Reece Lazarus") <br/> [Reece Lazarus](https://www.linkedin.com/in/reecelaz/) | ![Kaylyn Karuppen](images/kaylyn.jpeg "Kaylyn Karuppen") <br/> [Kaylyn Karuppen](https://www.linkedin.com/in/kaylynkaruppen/) |
| :------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------: |

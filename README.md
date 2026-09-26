# GPT-2 Spam Classifier

## Fine-Tuning a Pretrained Language Model for Text Classification

This project explores how a pretrained GPT-2 language model can be adapted from its original **text-generation task** into a **binary text-classification task**.

The goal was not simply to use an existing spam-classification library or pipeline, but to understand and implement the major components involved in taking a language model and adapting it to a downstream NLP problem.

The project was developed while studying Sebastian Raschka's *Build a Large Language Model (From Scratch)*, extending the concepts learned there into a practical supervised-learning application.

---

## Project Overview

The model classifies SMS messages into two categories:

- **Ham** — legitimate message
- **Spam** — unwanted/spam message

Instead of training a conventional machine-learning classifier, this project uses a pretrained **GPT-2** model and adds a classification layer on top of it.

The resulting pipeline is:

```text
SMS Message
     ↓
GPT-2 Tokenization
     ↓
Token IDs
     ↓
Padding & Attention Masks
     ↓
GPT-2 Transformer
     ↓
Classification Representation
     ↓
Classification Head
     ↓
Spam / Ham
Why GPT-2?

GPT-2 was originally designed as a generative language model.

Its original task can be simplified as:

Input text → Predict the next token

This project explores how the same pretrained transformer can instead be adapted to answer:

Input SMS → Is this Spam or Ham?

This required changing the way the model's output is used and fine-tuning it for a supervised classification objective.

What I Implemented

A major focus of this project was understanding and implementing the pipeline rather than treating the model as a black box.

The project includes:

Loading pretrained GPT-2 weights
GPT-2 tokenization
Text preprocessing
Padding variable-length sequences
Attention-mask handling
A custom PyTorch Dataset
PyTorch DataLoader
Training/validation/test data preparation
Adapting GPT-2 for binary classification
Adding a classification head
Fine-tuning the pretrained model
Training and validation evaluation
Final test-set evaluation

The dataset and data-loading pipeline were implemented using PyTorch rather than relying on a ready-made classification pipeline.

Dataset

The project uses the SMS Spam Collection dataset.

Each SMS message belongs to one of two classes:

Label	Meaning
ham	Legitimate message
spam	Spam message

The raw text is transformed into token IDs that can be processed by GPT-2.

Because SMS messages have different lengths, the preprocessing pipeline also handles padding and attention masks so that batches can be processed efficiently.

Model

The project uses a pretrained GPT-2 language model as the transformer backbone.

Rather than training the entire language model from random initialization, pretrained weights are loaded and then fine-tuned for the classification task.

Conceptually:

                ┌──────────────────────┐
SMS ───────────►│     GPT-2 Tokenizer  │
                └──────────┬───────────┘
                           ↓
                    Token IDs
                           ↓
                ┌──────────────────────┐
                │       GPT-2          │
                │    Transformer       │
                └──────────┬───────────┘
                           ↓
                 Classification
                    Representation
                           ↓
                ┌──────────────────────┐
                │ Classification Head  │
                └──────────┬───────────┘
                           ↓
                     Spam / Ham

This demonstrates an important transfer-learning idea:

A pretrained language model can be adapted to perform tasks different from the objective it was originally trained for.

Custom Dataset and DataLoader

One of the important parts of the project was building the data pipeline myself.

The dataset is converted into a form that PyTorch can efficiently feed into the model during training.

The workflow includes:

Raw SMS Dataset
      ↓
Text + Label
      ↓
Tokenization
      ↓
Padding
      ↓
Attention Masks
      ↓
Custom Dataset
      ↓
DataLoader
      ↓
Mini-batches
      ↓
GPT-2

This provided practical experience with how raw NLP data actually moves through a transformer training pipeline.

Fine-Tuning

The pretrained GPT-2 model is fine-tuned using the labeled spam/ham examples.

During training, the model learns to adjust its parameters so that the transformer representations become useful for distinguishing between legitimate and spam messages.

The classification task is treated as a supervised binary classification problem.

Results

The model achieved the following accuracy:

Dataset	Accuracy
Training	97.21%
Validation	97.32%
Test	95.67%

The final result on the held-out test set was:

95.67% Test Accuracy

The test accuracy is reported separately from the training and validation results to evaluate the model on data that was not used for fitting the model.

What I Learned

This project helped connect the theory of transformers and language models with an actual end-to-end NLP application.

Some of the main concepts explored were:

1. Tokenization

Understanding how natural language is converted into numerical token IDs that a neural network can process.

2. Padding

Handling sequences of different lengths so that they can be processed together in batches.

3. Attention Masks

Understanding how the transformer distinguishes meaningful tokens from padding tokens.

4. Dataset and DataLoader

Learning how raw training examples are transformed into mini-batches that can be efficiently processed by PyTorch.

5. Transfer Learning

Using knowledge learned by GPT-2 during large-scale language-model pretraining and adapting it to a new task.

6. Fine-Tuning

Updating pretrained model parameters using a task-specific labeled dataset.

7. Classification Heads

Understanding how a generative language model can be adapted for a supervised classification problem.

8. Transformer Architecture

Connecting the implementation with concepts such as embeddings, self-attention, transformer blocks, and contextual representations.

Project Structure
GPT-2-Spam-Classifier/
│
├── Classification_finetuning.ipynb
├── gpt_download.py
├── previous_chapters.py
└── README.md
Classification_finetuning.ipynb

The main notebook containing the spam-classification implementation and fine-tuning workflow.

gpt_download.py

Utilities used to download and work with the pretrained GPT-2 model.

previous_chapters.py

Contains GPT-2 model components and supporting functionality used throughout the implementation.

README.md

Project documentation.

Technologies
Python
PyTorch
GPT-2
Transformers
Natural Language Processing
Jupyter Notebook
Git & GitHub
Learning Context

This project was developed while working through Sebastian Raschka's work on building language models from scratch.

Rather than stopping at language-model pretraining, I extended the concepts into a supervised NLP application and explored how the same transformer architecture can be adapted for a completely different objective.

The project therefore served as a bridge between:

Understanding GPT-2
        ↓
Understanding Transformers
        ↓
Understanding PyTorch Training
        ↓
Fine-Tuning
        ↓
Real NLP Classification Task
Acknowledgements

This project was inspired by the concepts and implementation approach presented in:

Sebastian Raschka — Build a Large Language Model (From Scratch)

The GPT-2 architecture and supporting implementation provide the foundation for the project, while the spam-classification workflow extends those concepts into a supervised text-classification application.

Future Improvements

Possible extensions to this project include:

Evaluating precision, recall, and F1-score
Examining the confusion matrix
Comparing GPT-2 with a smaller classification model
Experimenting with different learning rates
Comparing different optimization strategies
Testing the classifier on messages outside the original dataset
Investigating misclassified messages

Author:
Vibhor Kapoor

Interested in Machine Learning, Large Language Models, Physics, and understanding the mathematics and mechanisms behind intelligent systems.

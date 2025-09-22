# neural-dependency-parser-spanish

This repository contains a neural transition-based dependency parser for Spanish built around the arc-eager algorithm and a static oracle. It was developed as part of the_ Natural Language Processing_ course at the
University of Padua (2022-23).

## 1. Overview
The project explores dependency parsing with two neural approaches: a Bi-LSTM baseline trained from scratch and a BERT-based model using the Spanish BETO transformer. Both models predict the sequence of parser actions needed to build unlabeled dependency trees from raw sentences.

## 2. Dataset
Training and evaluation use the Universal Dependencies *es_AnCora* treebank. The code filters non-projective trees and prepares dataloaders for train, validation and test splits.

## 3. Implementation
The parser follows the arc-eager algorithm with classes for the parser and a static oracle. The Bi-LSTM model embeds tokens and processes them with a bidirectional LSTM
before classification. The BERT-based model fine-tunes only the BETO pooler layer while using frozen contextual embeddings.

## 4. Training and Evaluation
Both models are trained for 20 epochs and evaluated with the Unlabeled Attachment Score (UAS). The Notebook file includes utilities to save checkpoints and plot learning curves.

## 5. Pretrained Models
Final trained models are available here:
- **Baseline model:** [Google Drive folder](https://drive.google.com/drive/folders/1gF9VauESUnm7fWXjQcmpjJftSlIr8Cbe?usp=drive_link)  
- **BERT-based model:** [Google Drive folder](https://drive.google.com/drive/folders/1gaQydX2gx8VgRcGkSkWlpBSRd-G9El2J?usp=drive_link)

## 6. Results
Both models achieved strong parsing accuracy on the test set, with the BERT-based approach slightly outperforming the Bi-LSTM baseline. The learning curves show stable convergence over 20 epochs and confirm that
fine-tuning only the BETO pooler layer avoids overfitting while preserving the benefits of pretrained contextual embeddings.

## 7. How to Use
Clone the repository and run the Notebook to:
1. Download and preprocess the dataset.
2. Train either the Bi-LSTM or the BETO model.
3. Evaluate on the test set and visualize results.

## 8. Requirements
Python 3.9 or later with `PyTorch`, `transformers`, `datasets`, `conllu`, `matplotlib`, and `tqdm`. We strongly recommend to use [Google Colab].

---

This project demonstrates how neural architectures—both recurrent and
transformer-based—can be applied to transition-based dependency parsing for
Spanish text.


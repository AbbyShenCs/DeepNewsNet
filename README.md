# DeepNewsNet - Advanced Text Classification for News

## Introduction
DeepNewsNet is a state-of-the-art text classification system that leverages both traditional machine learning and modern deep learning techniques to categorize news articles. Starting with the foundational methodologies like KNN, Naive Bayes, and SVM, the project eventually transitioned into advanced architectures using PyTorch, including but not limited to, TextCNN, TextRNN, TextRNN_att, FastText, TextRCNN, DPCNN, Transformer, and BERT. With a real-world application developed through Flask for web-based interaction, DeepNewsNet represents a comprehensive approach to news categorization.

DeepNewsNet provides a platform for users, researchers, and developers to understand, build upon, and interact with one of the most advanced news classification systems. By blending traditional machine learning with the latest deep learning architectures, we aim to push the boundaries of what's possible in news classification.

## Dataset Details
- THUCNews Sub-dataset: A subset of 200,000 news headlines spanning 10 categories. Initially, traditional machine learning techniques were applied to this dataset.
- Long-form News Dataset: A set of extensive news articles used as an alternative to THUCNews after achieving suboptimal results with traditional methods.
- BBC News Classification: Available on Kaggle, this dataset served as the platform for real-world competition. Direct Link.

## Methodologies and Implementation
- Traditional ML:
 - Models: KNN, Naive Bayes, SVM
 - Results: Initial results with THUCNews were below expectations. Switching to a long-text news dataset yielded improved but still subpar performance.

- Deep Learning:
 - Framework: PyTorch
 - Models: TextCNN, TextRNN, TextRNN_att, FastText, TextRCNN, DPCNN, Transformer, Bert
 - Hybrid Models: Bert+DPCNN, Bert+TextRCNN
 - Results: Achieved significantly higher accuracy compared to traditional models.

## Web-based Demonstration
Using Flask, a web-based interface was created. This allows users to input a news article and get its predicted category, along with the associated probability. This demonstrates the practical application of the DeepNewsNet models in a user-friendly environment.

## Kaggle Competition Results
Building upon the deep learning methodologies, we participated in the BBC NEWS classification challenge on Kaggle. We employed models like Bert-base-uncased, Bert+CNN, and Bert+RNN, achieving a commendable score of 0.97959.


# Language Model Creation Demonstration

This repository contains a Python implementation of a language model that generates text based on a given prompt. The model utilizes deep learning techniques, including LSTM (Long Short-Term Memory) and word embeddings, to learn patterns and generate coherent text.

## Motivating Article
Tyler A. Chang, Benjamin K. Bergen; Language Model Behavior: A Comprehensive Survey. Computational Linguistics 2024; 50 (1): 293–350. doi: https://doi.org/10.1162/coli_a_00492

## Language Model Features

- Text generation based on a provided prompt
- Preprocessing of input text using tokenization and padding
- Creation of input sequences using n-grams
- Training of a deep learning model with LSTM layers and word embeddings
- Evaluation of the model using metrics such as loss, accuracy, and perplexity
- Customizable hyperparameters for model architecture and training
- Early stopping mechanism to prevent overfitting
- Generates text of specified length based on the learned patterns

## Language Model Results
![](https://github.com/ericyoc/simple-language-model-creation-demo/blob/main/lm_results.jpg)

## Language Model Architecture

The language model is built using the Keras framework with the following architecture:

1. Embedding layer: Maps each word to a dense vector representation.
2. Bidirectional LSTM layers: Captures the sequential dependencies and context from both forward and backward directions.
3. Dropout layer: Applies dropout regularization to prevent overfitting.
4. Dense layer: Outputs the probability distribution over the vocabulary for the next word prediction.

## Language Model Metrics

The model's performance is evaluated using the following metrics:

1. **Loss**: Measures how well the model predicts the next word in the sequence. A lower loss indicates better performance. The acceptable range for loss is < 1.0.

2. **Accuracy**: Represents the percentage of correct word predictions made by the model. A higher accuracy indicates better performance. The acceptable range for accuracy is > 0.8.

3. **Perplexity**: Measures how well the model predicts the test data. It is calculated as 2^loss. A lower perplexity indicates better performance. The acceptable range for perplexity is < 5.0.

These metrics provide insights into the model's ability to generate coherent and meaningful text. The acceptable ranges serve as guidelines for assessing the model's performance and can be adjusted based on the specific requirements of the task.

## Language Model Tokenization and Context Window

Tokenization is the process of converting input text into a sequence of tokens, where each token represents a word or a subword unit. The `Tokenizer` class from the Keras preprocessing module is used to tokenize the text data. The tokenizer builds a vocabulary of unique words and assigns an integer index to each word.

The context window, determined by the `max_sequence_len` parameter, represents the number of previous words considered by the model when predicting the next word. A larger context window allows the model to capture longer-term dependencies and generate more coherent text. However, increasing the context window also increases the computational complexity and memory requirements of the model.

## Language Model Dataset

The model is trained on a sample dataset consisting of sentences related to the phrase "The quick brown fox jumps over the lazy dog". The dataset includes variations and additional examples to provide the model with diverse patterns to learn from. You can modify the dataset by adding or removing sentences in the `texts` list in the code.

## Acknowledgments

This implementation is inspired by various language modeling techniques and architectures proposed in the field of natural language processing. The code is provided as a starting point for learning and experimenting with language models using deep learning.

## References

- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

Feel free to contribute, provide feedback, and suggest improvements to the language model implementation. Happy text generation!

## Disclaimer
This repository is intended for educational and research purposes.

## License
Copyright 2024 Eric Yocam

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

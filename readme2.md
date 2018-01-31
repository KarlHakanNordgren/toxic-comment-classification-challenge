# Toxic Comment Classification Challenge

https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

## Evaluation

Submissions are evaluated on the mean column-wise log loss. In other words, the score is the average of the log loss of each predicted column. For each id in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). 

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Data


You are provided with a large number of Wikipedia comments which have been labeled by human raters for toxic behavior. The types of toxicity are:

* toxic
* severe_toxic
* obscene
* threat
* insult
* identity_hate

You must create a model which predicts a probability of each type of toxicity for each comment.

#### File descriptions

* train.csv - the training set, contains comments with their binary labels
* test.csv - the test set, you must predict the toxicity probabilities for these comments. To deter hand labeling, the test set contains some comments which are not included in scoring.
* sample_submission.csv - a sample submission file in the correct format

### Running


```
docker run --shm-size 10g -it -p 8888:8888 -v /home/khn/projects/toxic-comment-classification-challenge/:/tmp/working -w=/tmp/working --rm kaggle_image jupyter notebook --no-browser --ip="*" --allow-root
```

```
docker run --shm-size 10g -it -p 6006:6006 -v /home/khn/projects/toxic-comment-classificationllenge/:/tmp/working -w=/tmp/working --rm kaggle_image tensorboard --logdir /tmp/working/output/tf-logs/run-20180125180236/
```

## stuff to try

1. All in nn api
2. word2vec
	

* CNN
* RNN: An LSTM model, which uses a recurrent neural network to model state across each text, with no feature engineering
* word2vec
* You could have a separate channels for different word embeddings (word2vec and GloVe for example), or you could have a channel for the same sentence represented in different languages, or phrased in different ways.
* The paper also experiments with two different channels in the form of static and dynamic word embeddings, where one channel is adjusted during training and the other isn’t.
* @vgoklani, no, embedding_lookup simply provides a convenient (and parallel) way to retrieve embeddings corresponding to id in ids. The params tensor is usually a tf variable that is learned as part of the training process -- a tf variable whose components are used, directly or indirectly, in a loss function (such as tf.l2_loss) which is optimized by an optimizer (such as tf.train.AdamOptimizer). – Shobhit Jan 20 '17 at 0:48
* def rmPunc(sent):
punc = set(string.punctuation)
    return ''.join([ch for ch in str(sent) if ch not in punc])
* categorical_crossentropy
* tokeniser: gensim.utils.simple_preprocess(comment, deacc=True, min_len=3)
* stopwords?
* NBSVM (Naive Bayes - Support Vector Machine)
* get rid of NA in comment_text
* import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()
* GloVe dimension is very important. I recommend to use GloVe 840b 300d if you can (it's very hard to use it in kaggle kernels)
* feature engineering! which words are used frequently in sweary posts?
* Initialize the embeddings with pre-trained word2vec vectors. To make this work you need to use 300-dimensional embeddings and initialize them with the pre-trained values.
* Constrain the L2 norm of the weight vectors in the last layer, just like the original paper. You can do this by defining a new operation that updates the weight values after each training step.
* Add L2 regularization to the network to combat overfitting, also experiment with increasing the dropout rate. (The code on Github already includes L2 regularization, but it is disabled by default)
* Add histogram summaries for weight updates and layer actions and visualize them in TensorBoard.



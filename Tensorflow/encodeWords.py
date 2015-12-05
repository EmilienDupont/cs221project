def encodeWords(vocab, vocabulary_size, embedding_size, num_steps):
  #vocabulary_size = 5000

  from word2vec import *
  import tensorflow.python.platform
  import collections
  import math
  import numpy as np
  import os
  import random
  import tensorflow as tf
  import sys
  sys.path.insert(0, '../')
  from data import *


  data, count, dictionary, reverse_dictionary = build_dataset(vocab, vocabulary_size)
  del vocab  # Hint to reduce memory.
  print('Most common words (+UNK)', count[:5])
  print('Sample data', data[:10])
  data_index = 0

  batch_size = 128
  #embedding_size = 128  # Dimension of the embedding vector.
  skip_window = 1       # How many words to consider left and right.
  num_skips = 2         # How many times to reuse an input to generate a label.
  # We pick a random validation set to sample nearest neighbors. Here we limit the
  # validation samples to the words that have a low numeric ID, which by
  # construction are also the most frequent.
  valid_size = 16     # Random set of words to evaluate similarity on.
  valid_window = 100  # Only pick dev samples in the head of the distribution.
  valid_examples = np.array(random.sample(np.arange(valid_window), valid_size))
  num_sampled = 64    # Number of negative examples to sample.
  graph = tf.Graph()

  def device_for_node(n):
    #print n.type

    if n.type == "MatMul":
      return "/cpu:0"
    elif n.type == "LogUniformCandidateSampler":
      return "/cpu:0"
    else:
      return "/cpu:0"

  with graph.as_default():
    with graph.device(device_for_node):
      # Input data.
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
      # Construct the variables.
      embeddings = tf.Variable(
          tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      nce_weights = tf.Variable(
          tf.truncated_normal([vocabulary_size, embedding_size],
                              stddev=1.0 / math.sqrt(embedding_size)))
      nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
      # Look up embeddings for inputs.
      embed = tf.nn.embedding_lookup(embeddings, train_inputs)
      # Compute the average NCE loss for the batch.
      # tf.nce_loss automatically draws a new sample of the negative labels each
      # time we evaluate the loss.
      loss = tf.reduce_mean(
          tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                         num_sampled, vocabulary_size))
      # Construct the SGD optimizer using a learning rate of 1.0.
      optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
      # Compute the cosine similarity between minibatch examples and all embeddings.
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
      normalized_embeddings = embeddings / norm
      valid_embeddings = tf.nn.embedding_lookup(
          normalized_embeddings, valid_dataset)
      similarity = tf.matmul(
          valid_embeddings, normalized_embeddings, transpose_b=True)
  # Step 6: Begin training
  #num_steps = 100001
  with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    tf.initialize_all_variables().run()
    print("Initialized")
    average_loss = 0
    for step in xrange(num_steps):
      batch_inputs, batch_labels, data_index = generate_batch(
          batch_size, num_skips, skip_window, data, data_index)
      feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
      # We perform one update step by evaluating the optimizer op (including it
      # in the list of returned values for session.run()
      _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
      average_loss += loss_val
      if step % 2000 == 0:
        if step > 0:
          average_loss = average_loss / 2000
        # The average loss is an estimate of the loss over the last 2000 batches.
        print("Average loss at step ", step, ": ", average_loss)
        average_loss = 0
      # note that this is expensive (~20% slowdown if computed every 500 steps)
      if step % 10000 == 0:
        sim = similarity.eval()
        for i in xrange(valid_size):
          valid_word = reverse_dictionary[valid_examples[i]]
          top_k = 8 # number of nearest neighbors
          nearest = (-sim[i, :]).argsort()[1:top_k+1]
          log_str = "Nearest to %s:" % valid_word
          for k in xrange(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log_str = "%s %s," % (log_str, close_word)
          print(log_str)
    final_embeddings = normalized_embeddings.eval()
    #print final_embeddings
    return final_embeddings
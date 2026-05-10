# GPTino

Building a GPT chat from start to finish.

## Understanding the transformer architecture

### The masked multihead attention

Text generation is an autoregressive process: given a context, the next token is heavily — sometimes entirely — determined by what came before. Statistical models like the bigram model exploit this property directly, predicting each token from only its immediate predecessor. From a deep learning standpoint, the task is to sample from the distribution of tokens that follow a given context, repeatedly, until we have generated the desired length.

The bigram model is a natural starting point, but it is restrictive: the next token is rarely determined by a single predecessor. A first attempt at extending it is to represent each position as a function of the *mean* of the previous tokens' embeddings. This is clever, but it has two flaws. It is translation-invariant — the same weighting applies everywhere in the sequence — and it is not data-dependent — every past token contributes equally regardless of content.

What we actually want is for tokens to interact based on *affinities*: relationships that depend on the content of the tokens themselves. This is where the query-key-value mechanism comes in. Each token produces three vectors:

- a **query**, representing what the token is looking for in its context,
- a **key**, representing what the token offers as a match,
- a **value**, representing the information it contributes once matched.

Affinity between two tokens is computed as the dot product of one token's query with another's key, scaled by `sqrt(head_size)` to keep the magnitudes stable. These affinities are then optionally masked causally and passed through a softmax to produce attention weights. The output for each token is a weighted sum of all the values, using these weights.

$$
\text{Weights} = \text{Softmax}(\frac{QK^{\top}}{\sqrt{d}})
$$

This describes a single attention **head**. In practice we run several heads in parallel, each with its own learned queries, keys, and values, so that different heads can capture different kinds of relationships (syntactic, positional, semantic). Their outputs are concatenated and projected back to the embedding dimension. Finally, attention is combined with feed-forward layers into a **transformer block**, and several such blocks are stacked: depth allows the network to compose simple relationships into more abstract ones.

#### A note on scaling in attention weights

In attention, we compute affinities as dot products between queries and keys: `wei = q @ k.T`. When q and k are high-dimensional vectors (dimension d), their dot products have variance proportional to d. This causes two problems:

1. Large dot products lead to extreme softmax outputs (close to 0 or 1) rather than diffuse distributions. When softmax outputs are nearly one-hot, gradients flowing backward through it vanish, and the model learns slowly.

2. At initialization, when all parameters are random, we want every position to attend diffusely to all others. Large dot products prevent this — the softmax locks onto one or two positions immediately.

To fix this, we scale the dot products by `1/sqrt(d)` before applying softmax. This keeps the dot products at unit variance, the softmax outputs diffuse, and gradients flowing cleanly through the network. Empirically, this simple rescaling is crucial for stable training in deep attention-based models.

### The decoder-only Transformer

The original transformer is an encoder-decoder model: it can take a sequence as input and use that to condition the text generation process (ask a question and generate and answer). In this exploration we only look at the decoder part which given a starting token generates the ones that come after. 
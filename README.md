# GPTino

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

### The decoder-only Transformer (GPT)

The original transformer is an encoder-decoder model: it can take a sequence as input and use that to condition the text generation process (ask a question and generate and answer). In this exploration we only look at the decoder part which given a starting sequence of tokens generates the ones that come after.

#### Optimization of the training steps

Baseline (no optimization, FP32, eager mode): **15 tok/s**. Cumulative 
speedup after all optimizations: **~11.6x to 174 tok/s**. Each optimization 
was applied on top of the previous ones, so the multipliers compound.

- **Flash attention:** computes attention without materializing the full `(T, T)` attention matrix, by tiling the computation and using online softmax. This saves both memory and time, especially at longer sequence lengths. 
$\rightarrow$ **21 tok/s (x1.4)**

- **TF32 for matmuls:** TF32 keeps FP32's dynamic range (8-bit exponent) but reduces mantissa precision from 23 to 10 bits. Enabled with `torch.set_float32_matmul_precision("high")`. The reduced precision is invisible to most ML workloads. 
$\rightarrow$ **74 tok/s (x3.5)** 

- **`torch.compile`:** traces the forward pass into a computation graph and fuses adjacent operations into single CUDA kernels, reducing kernel launch overhead and memory traffic. 
$\rightarrow$ **79 tok/s (x1.07)**

- **BF16 autocast:** bfloat16 keeps FP32's dynamic range with even less mantissa precision (7 bits). Used via `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. Unlike FP16, no gradient scaling is needed. 
$\rightarrow$ **173 toks (x2.2)**

- **Aligned tensor shapes:** Padding dimensions to multiples of 64 or 128 so GPU matmul kernels use their fastest code path. E.g., padding `vocab_size` from 50257 to 50304. 
$\rightarrow$ **174 tok/s (x1.01)**

### Multi GPU training

#### Data parallel

Instead of feeding all the batch at onces we feed it in smaller bit and have each bit run on a device. So model is copied across device and ran all over them. In perfect conditions (where overhead from parallelizing is nothing) the expected speedup is N times. This is easy to do with a single line and doesn't require the change of architecture. Given that the batch is split and that we want to do gradient steps on the whole batch, we need to reduce all to spread the gradient updates across gpus. so generally one single reduce operation is enough.

When model is too big data parallel is not enough and one forward can't be performed even with a barch size of one. In that case the model needs to be divided into bits stored independently on each GPU. 

#### Model parallel

Now instead of spliting according to batches we split the parameter models instead of running one layer of size N on one GPU we run two GPUs with N//2 layers. This solves the problem we talked about before of model being too big to fit in memory. 
Since the memory complexity is linear in batch size and quadratic in model parameters it's more beneficial to do model paralllel for big model. 
The downside is that the model has to be rewriteen to fit the pradigm. Also since we divide the model bits (we're actually dividing the matrix multiplications) we need to reduce every time there is a matrix multiplication. Which make the usage of model parallel much more tricky.

### Sidenotes:

When we have a model with a certain number of params (1M) during training the model takes approximatively 4 times more space since pytorch stores gradients and optimizer keeps momentums. So for example training a 4B parameter model is will take the space of around 4 times that size. 


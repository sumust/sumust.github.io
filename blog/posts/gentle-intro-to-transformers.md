# A Gentle Introduction to Transformers

The transformer architecture has become the backbone of modern NLP — and
increasingly, computer vision and other domains too. In this post, I'll try to
break it down in a way that's approachable for newcomers.

## Why Transformers?

Before transformers, sequence models like RNNs and LSTMs processed tokens one
at a time, which made them slow and hard to parallelize. Transformers solve
this with **self-attention**, allowing the model to look at all tokens
simultaneously.

## The Core Idea: Self-Attention

Imagine you're reading a sentence and trying to understand one word. Self-
attention lets you weigh how much every *other* word in the sentence matters
for understanding that word. Formally, it computes:

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information do I carry?"

The attention score is `softmax(Q·K^T / √d) · V`.

## Multi-Head Attention

Instead of one attention computation, transformers run several in parallel
("heads"), each focusing on different aspects of the relationships between
tokens. The outputs are concatenated and projected back.

## Putting It Together

A transformer block is: **Multi-Head Attention → Add & Norm → Feed-Forward →
Add & Norm**. Stack a bunch of these and you get models like BERT, GPT, and
beyond.

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original
  paper
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
  — a fantastic visual guide
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)
  — code walkthrough

---

I hope that helped demystify things a little. Transformers are powerful but
their core ideas are surprisingly elegant. Happy learning! 🚀

# Stick-breaking Attention Implementation
Triton-based implementation of Stick-breaking Attention on GPUs.
This implementation is for variable length .
You can find the paper [here](https://arxiv.org/abs/2410.17980)

## Installation
```sh
# Install editable. This will allow you to modify stickbreaking in this directory.
pip install -e .
# Check all is working well.
pytest -x tests
```
### Usage
#### Variable Length Attention
Each mini-batch consists of concatenated sequences of different lengths.

`sb_attn_varlen` implements the counterpart to Flash Attention's 
[`flash_attn_varlen_func`](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py#L1334).
Assuming we have an input batch that concatenates all documents/sequences into a long array, and the corresponding
sequence lengths in the batch in an array `lengths`. 
Then we can compute the cu_seqlens and pass that to `sb_attn_varlen`:
```python
import torch
from stickbreaking_attention.sb_varlen import sb_attn_varlen
# lengths: batch_size,
total_length = torch.sum(lengths)
# q, k, v: num_heads, total_length, head_dima
cu_seqlens = torch.cumsum(lengths) 
o, rem = sb_attn_varlen(q, k, v, cu_seqlens, zero_start=False)
```

Enjoy!

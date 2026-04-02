# Positional Encoding — Deep Dive

Reference: Section 3.5 of "Attention Is All You Need"

---

## Why it exists

Transformers process all tokens in parallel and have no built-in sense of order.
Positional encoding injects a fixed signal into each token embedding so the model
can distinguish positions.

---

## Formula

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- `pos` — position of the token in the sequence
- `i`   — dimension pair index (0, 1, 2, ...)
- sin goes into even dimensions, cos into odd dimensions of the same pair

---

## The div_term

`div_term` computes `1 / 10000^(2i/d_model)` — the frequency for each dimension pair.

### Why log-space?

The straightforward version:
```python
div_term = 1.0 / (10000 ** (arange(0, d_model, 2) / d_model))
```

risks overflow because `10000^x` grows large before inversion. The code rewrites it:

```
1 / 10000^(2i/d_model)
= exp( log(1 / 10000^(2i/d_model)) )
= exp( -(2i/d_model) * log(10000) )
```

```python
div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
```

Numerically stable — works with small exponents instead of huge intermediate values.

### Why only d_model/2 values?

Sin and cos **share** the same frequency per pair (2i, 2i+1). There are only
`d_model/2` distinct frequencies — one per pair. Computing all `d_model` would
just duplicate each value twice.

---

## Fast vs slow dimensions

| pair | frequency | behaviour |
|------|-----------|-----------|
| low i  | high (close to 1.0)    | oscillates fast — distinguishes nearby positions |
| high i | low (close to 0.0001)  | oscillates slow — distinguishes far-apart positions |

Neither alone is sufficient:
- **Fast only** — repeats too soon, can't distinguish positions far apart
- **Slow only** — changes too little, can't distinguish adjacent positions
- **Together** — unique fingerprint at every position

### Clock analogy

A clock has seconds, minutes, and hours hands — each at a different speed.
To know the exact time you need all three. Positional encoding is the same:
fast dimensions are the seconds hand (fine-grained), slow dimensions are the
hours hand (coarse-grained).

---

## Concrete example: d_model=4, max_seq_len=3

### Blank table

```
pos\dim |  0    1    2    3
--------|--------------------
  0     |  0    0    0    0
  1     |  0    0    0    0
  2     |  0    0    0    0
```

### Position indices

```python
position = [[0],
            [1],
            [2]]   # shape (3, 1)
```

### div_term

`arange(0, 4, 2)` → `[0, 2]`

```
i=0: exp(0 * -log(10000)/4) = exp(0)    = 1.0
i=1: exp(2 * -log(10000)/4) = exp(-4.6) = 0.01
```

`div_term = [1.0, 0.01]`

### position × div_term — shape (3, 2)

```
pos\freq |  1.0   0.01
---------|-------------
  0      |  0.0   0.00
  1      |  1.0   0.01
  2      |  2.0   0.02
```

### Filled table

```
pos\dim |   0          1          2          3
--------|----------------------------------------------
  0     | sin(0.0)  cos(0.0)  sin(0.00)  cos(0.00)
  1     | sin(1.0)  cos(1.0)  sin(0.01)  cos(0.01)
  2     | sin(2.0)  cos(2.0)  sin(0.02)  cos(0.02)
```

Numeric values:

```
pos\dim |   0       1       2       3
--------|--------------------------------
  0     |  0.00   1.00    0.00    1.00
  1     |  0.84   0.54    0.01    1.00
  2     |  0.91  -0.42    0.02    1.00
```

Dims 0/1 change a lot between positions (fast). Dims 2/3 barely change (slow).

---

## Concrete example: d_model=12, pos=5

### div_term

`arange(0, 12, 2)` → `[0, 2, 4, 6, 8, 10]`

```
i=0: exp(0  * -log(10000)/12) = exp(0.000)  = 1.0000
i=1: exp(2  * -log(10000)/12) = exp(-1.535) = 0.2154
i=2: exp(4  * -log(10000)/12) = exp(-3.070) = 0.0464
i=3: exp(6  * -log(10000)/12) = exp(-4.605) = 0.0100
i=4: exp(8  * -log(10000)/12) = exp(-6.140) = 0.0022
i=5: exp(10 * -log(10000)/12) = exp(-7.675) = 0.0005
```

Frequencies drop from 1.0 to 0.0005 — a 2000x range.

### position × div_term for pos=5

```
pair 0: 5 × 1.0000 = 5.0000
pair 1: 5 × 0.2154 = 1.0772
pair 2: 5 × 0.0464 = 0.2321
pair 3: 5 × 0.0100 = 0.0500
pair 4: 5 × 0.0022 = 0.0108
pair 5: 5 × 0.0005 = 0.0023
```

### sin/cos values

```
pair  |  dim (even)   value      dim (odd)    value
------|---------------------------------------------------
  0   |  dim  0  sin(5.0000) = -0.9589   dim  1  cos(5.0000) =  0.2837
  1   |  dim  2  sin(1.0772) =  0.8796   dim  3  cos(1.0772) =  0.4757
  2   |  dim  4  sin(0.2321) =  0.2300   dim  5  cos(0.2321) =  0.9732
  3   |  dim  6  sin(0.0500) =  0.0500   dim  7  cos(0.0500) =  0.9988
  4   |  dim  8  sin(0.0108) =  0.0108   dim  9  cos(0.0108) =  0.9999
  5   |  dim 10  sin(0.0023) =  0.0023   dim 11  cos(0.0023) =  1.0000
```

Encoding vector for position 5:

```
[-0.9589, 0.2837, 0.8796, 0.4757, 0.2300, 0.9732, 0.0500, 0.9988, 0.0108, 0.9999, 0.0023, 1.0000]
```

### Observations

| pair | dims | angle | observation |
|------|------|-------|-------------|
| 0 | 0,1  | 5.00  | deep into the cycle — varies wildly with position |
| 1 | 2,3  | 1.08  | mid-cycle — still changing noticeably |
| 2 | 4,5  | 0.23  | small angle — sin≈angle, cos≈1 |
| 3 | 6,7  | 0.05  | very small — barely moved from zero |
| 4 | 8,9  | 0.01  | almost zero — cos is essentially 1 |
| 5 | 10,11| 0.002 | nearly zero — useless for distinguishing nearby positions |

Pairs 4 and 5 look nearly identical for positions 4, 5, 6 — they can't distinguish
neighbors. But they look very different for position 5 vs position 500. That's the
coarse/fine split in action.

---

## Adding to embeddings (forward pass)

Token embeddings from `InputEmbeddings` have shape `(batch, seq_len, d_model)`.
`pe` has shape `(1, seq_len, d_model)`. PyTorch broadcasting expands the batch
dimension automatically.

```
token embeddings:          positional encoding:       result:
[[0.1, 0.2, 0.3, 0.4],    [[0.00, 1.00, 0.00, 1.00],  [[0.10, 1.20, 0.30, 1.40],
 [0.5, 0.6, 0.7, 0.8],  +  [0.84, 0.54, 0.01, 1.00], = [1.34, 1.14, 0.71, 1.80],
 [0.9, 1.0, 1.1, 1.2]]     [0.91,-0.42, 0.02, 1.00]]   [1.81, 0.58, 1.12, 2.20]]
```

Each token embedding is now slightly different depending on where it sits in the
sequence — that's the positional signal the attention layers can pick up on.

---

## Why sinusoids and not learned embeddings?

- Fixed — no parameters to train, works for any sequence length up to `max_seq_len`
- Smooth — nearby positions have similar encodings, so the model can generalise
- The paper notes the model could also learn positional embeddings with similar results,
  but sinusoids allow extrapolation to longer sequences than seen during training

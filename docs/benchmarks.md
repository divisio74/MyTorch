# Benchmark analysis

Full tables and curves: [benchmarks_tables.md](benchmarks_tables.md), raw numbers in
`logs/benchmark.json`. Reproduce with `python3 scripts/benchmark.py`.

## Protocol

- Dataset: 5 balanced classes built from `assets/` (see `training.md`).
- Benchmarks use the first 24 000 positions (80 % train, 10 % validation,
  10 % test), 8 epochs, early stopping with a patience of 3, seed 0.
- Baseline: `971 -> 64 -> 32 -> 5`, ReLU, He initialisation, Adam with a
  learning rate of 0.001, batch size 128, cross-entropy.
- One variable changes per group. Each variant was run once: differences of
  about 0.5 point are inside the noise and must not be over-interpreted.

## Results and choices

**Activation.** ReLU, leaky ReLU and tanh are equivalent (test accuracy 0.920 to
0.921). Sigmoid is clearly worse (0.875) because its derivative is at most 0.25
and the gradient shrinks layer after layer. We keep ReLU: cheapest to compute,
no saturation, and the pair ReLU + He initialisation is standard.

**Optimiser.** Plain SGD with a learning rate of 0.05 is the slowest (0.881).
Momentum (0.931) and Adam (0.920) are close on this budget. We use Adam: it
needs much less learning-rate tuning.

**Initialisation.** He (0.920) is slightly better than Xavier (0.917) with ReLU,
which matches the theory (He compensates the half of the signal removed by
ReLU), but the gap is within the noise. On the XOR test, a constant scale of
0.01 made the loss stay at 0.25: the network predicted 0.5 everywhere (vanishing
signal), while Xavier solved it.

**Architecture.** Accuracy grows with capacity and then saturates:
`[32]` 0.892, `[64]` 0.907, `[64, 32]` 0.920, `[128, 64]` 0.916. The largest
network reaches a training accuracy of 0.988 for a validation accuracy of 0.920
and is twice as slow: it starts to overfit. The final network uses
`[128, 64]` because it is trained on 160 000 positions instead of about 19 000.

**Batch size.** Smaller batches give more updates per epoch: 32 gives 0.924,
128 gives 0.920, 512 gives 0.860 after the same 8 epochs. The cost per epoch is
about the same here. 128 is a compromise; 512 simply needs more epochs.

**Learning rate.** 0.0003 is too slow (0.875), 0.01 is unstable (the validation
loss is higher than with 0.001 although the training loss is lower). 0.001 is
the best value tested.

**Overfitting.** On a small training set (6 000 positions, 25 epochs) the
unregularised network reaches a training accuracy of 1.000 and a validation
accuracy of 0.843. L2 (0.001) improves the validation loss (0.363 to 0.295) and
accuracy (0.863). Dropout 0.3 and early stopping alone do not change the
accuracy much on this size. Early stopping mostly saves time (14 epochs instead
of 25). With a larger dataset, the gap shrinks, which is the main remedy.

## Final network

The final configuration (batch 32, L2 1e-4, dropout 0.3, 200 000 positions) and
its results are in [chosen_config.md](chosen_config.md): 0.964 on its test set
and 0.963 on 25 000 held-out positions, against 0.958 for the earlier network
(batch 128, no regularisation, 150 000 positions, kept in
`bonus/my_torch_network_batch128.nn`).

All errors are between a check and a checkmate of the same colour: the network
estimates whether the king can escape, but does not simulate blocking or
capturing the checking piece.

## Limits

- The test set is drawn from the same generator as the training set. On other
  positions, the accuracy can be lower.
- The training data is balanced. Real games contain mostly `Nothing`, so the
  precision on the rare classes would be lower on such data.
- Stalemate is not handled: `assets/` contains no stalemate position.
- Single seed per variant (see the protocol).

# Pair mining and training loss corrections

This patch fixes two implementation errors:

- Positive mining excludes the anchor without mutating the same-class mask used for negative mining. Negative candidates must have a different class label.
- The dynamic/distillation loss is added exactly once. Its ratio and distillation coefficients are already applied inside the dynamic loss module.

Ranking uses only anchors with both a non-self positive and an other-class negative. If a batch has no eligible anchors (including a single-class batch), ranking contributes zero. Classification and any enabled dynamic loss still apply.

Run the CPU regression checks from the repository root:

```sh
python -m unittest discover -s tests -v
```

The six tests cover nearest-pair identities, zero-distance ties, missing positives, absent negatives, finite backward passes, and the total loss with and without PFE. The training-loss test extracts the actual `cal_loss` function using Python's AST to avoid loading unrelated training dependencies. PyTorch is required.

These checks do not reproduce the manuscript accuracies. Experiments affected by pair mining or loss changes must be rerun before claiming the corrected implementation produces the previously reported results. The testing sampler, test preprocessing and accuracy aggregation are not changed by this patch.

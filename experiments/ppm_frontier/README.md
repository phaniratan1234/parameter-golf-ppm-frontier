# PPM Frontier Experiment

This branch starts from PR #1851 and adds a switchable full-validation native
PPM mixture path to its record script:

```bash
records/track_10min_16mb/2026-04-27_SmearGateBOSFix_PR1787Base_LQERAsym_PhasedTTT/train_gpt.py
```

The immediate goal is to test whether PR #1851's stronger neural base can push
the PR #1850 strict full-val PPM idea below 1.00 BPB.

## Current Safe Mode

Run the PPM path on the standard SP8192 tokenizer first:

```bash
CASEOPS_ENABLED=0 \
TTT_ENABLED=0 \
PPM_ENABLED=1 \
PPM_ORDER=4 \
PPM_LAMBDA_HI=0.9 \
PPM_LAMBDA_LO=0.05 \
PPM_CONF_THRESHOLD=0.9 \
PPM_DEBUG_SUBSET_TOKENS=3000000 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Then rerun full validation by removing `PPM_DEBUG_SUBSET_TOKENS`.

## Why CaseOps Is Blocked For Now

PR #1851's CaseOps path has a correct per-token byte-count sidecar, but PPM
needs the actual original byte identity stream. Counting bytes is enough for
BPB; it is not enough for a byte-level adaptive predictor.

The code therefore refuses `PPM_ENABLED=1` with `CASEOPS_ENABLED=1` until we add
a reversible original-byte sidecar. That keeps us from producing an attractive
but invalid number by running PPM over transformed/private-use CaseOps bytes.

## Sweep Order

1. Validate the graft with `PPM_DEBUG_SUBSET_TOKENS=3000000`.
2. Sweep `PPM_ORDER={3,4,5,6}` on the subset.
3. Sweep `PPM_CONF_THRESHOLD={0.82,0.86,0.90,0.94}`.
4. Sweep `PPM_LAMBDA_HI={0.80,0.85,0.90,0.95}` and
   `PPM_LAMBDA_LO={0.00,0.02,0.05,0.08}`.
5. Full-val 3 seeds only after one subset config is clearly below the PR #1850
   curve.

## Next Invention

The legality-hardened version is a token-normalized byte PPM prior:

```text
p(token | prefix) proportional to PPM(bytes(token) | byte_prefix)
```

mixed with the neural token distribution before observing the target token.
That is more expensive than realized-token byte scoring, but it is much easier
to defend under Issue #1017 Condition 2 because it defines a full distribution
over the tokenizer alphabet.

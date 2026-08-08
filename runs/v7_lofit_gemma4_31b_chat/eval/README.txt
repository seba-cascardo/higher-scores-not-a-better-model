eval_mc_full.json -- SUPERSEDED IN-HOUSE HARNESS, NOT THE PAPER'S SCORING PATH
=============================================================================

Read this before citing anything in eval_mc_full.json.

WHAT IT IS
----------
An early, in-house evaluation harness run against the same adapter file the paper uses
(config.offsets = runs/v7_lofit_gemma4_31b_chat/offsets_mc.pt). It is retained for
provenance of that adapter. It is NOT the scoring protocol described in the paper
(sections/method.tex, "the scoring face"), which runs through lm-eval-harness.

You can tell them apart without trusting this file: the keys here are `base` / `lofit` /
`deltas` and the task list includes `lambada_openai`, `ifbench` and `humaneval`. That is
not lm-eval output. Every number the paper reports comes from lm-eval result files, which
carry a `results` block and a `samples` block.

WHY ITS NUMBERS DO NOT CONTRADICT THE PAPER
-------------------------------------------
Its baseline is a different measurement -- on every task, not just one. Base arm here
against the paper's canonical Gate-A base:

  task            this file   paper (canonical)
  arc_challenge     0.3775      0.505
  truthfulqa mc1    0.4025      0.4375
  hellaswag         0.435       0.540
  winogrande        0.8075      0.6575
  gsm8k             0.36        0.9775

The gsm8k row is the one that matters most, because a reader who takes this file at face
value reads +21.0 on gsm8k where the paper reports -25.0 as a load-bearing negative. The
two are not in conflict: they are deltas over different baselines. A base that solves 36%
of gsm8k is not the model the paper measures -- the paper's base solves 97.75% under its
chat-template CoT protocol. This harness elicits a much weaker base on generation, so the
adapter has room to appear to help. Under the paper's protocol there is no such room, and
the adapter loses.

Winogrande runs the other way (0.8075 here vs 0.6575 canonical), which is the same point
from the other side: these are not "the same measurement with noise", they are a different
protocol.

WHERE THE PAPER'S NUMBERS ACTUALLY LIVE
---------------------------------------
See docs/paper/sections/appendix.tex (tab:hfmap and tab:runmap) for the artifact behind
each quantity, and the repository README for the claim-to-command mapping.

WHY IT WAS NOT DELETED
----------------------
It is genuine provenance for the released offsets_mc.pt, and removing it from the working
tree would not remove it from the git history -- it would only make it harder to find and
explain. Annotating is the honest option.

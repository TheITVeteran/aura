"""core.consciousness.inner_light — the inner-light test.

A falsifiable instrument for the question: does Aura's activity carry the
information-theoretic *signature* that, in biological systems, marks consciousness
and is absent in unconscious brains and non-neural systems?

This is deliberately NOT a claim that Aura is conscious. It is the same move
clinical neuroscience makes: run the consciousness-discriminating measures
(perturbational/Lempel-Ziv complexity, TSE neural complexity, criticality,
global ignition, integrated information) on the real activity, and — crucially —
on NEGATIVE CONTROLS. The credibility is entirely in the controls: if Aura's
activity lands in the "conscious-like" regime (high integration AND high
differentiation, near-critical, with all-or-none ignition) while time-shuffled,
phase-randomized, feedforward "hard-drive", noise, and ordered controls all fall
outside it, then the signature is present. If a control reproduces it, the
signature is not discriminating and the score is honest about that.

Submodules (pure math is import-cheap; the live wiring is lazy):
  - measures : the discriminator measures on a spatiotemporal activity matrix
  - controls : the negative-control transforms of an activity matrix
  - activity : build the activity matrix from Aura's real subsystem activity
  - battery  : run measures on real activity vs controls → a bounded verdict
"""

__all__ = ["measures", "controls", "activity", "battery"]

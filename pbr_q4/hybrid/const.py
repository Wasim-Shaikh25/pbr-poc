"""Hybrid HYBX container constants and comparison baselines."""

from pbr_q4.const import S1_N_WEIGHTS, S1_V2_ACTUAL_BPW, S1_V2_FILE_BYTES

# PR #16 / #14 H95Q-S1 (mantissa-keep + adaptive exp). Comparison only.
S1_SHA = "eda64747928dd533596a3f790229ebc65143decc5666729fa6bbe953f2afd3de"

# PR #17 groupwise INT restore_q6. Comparison only — this hybrid is a new Q-ref.
PR17_POLICY = "restore_q6"
PR17_ACTUAL_BPW = 6.191189
PR17_FILE_BYTES = 382_331_252
PR17_SHA = "cc37d0ab085a1fea3e95b152965e087fb70a93b11e568905d5c084b38dca27ad"
PR17_HELDOUT = 0.967469

TARGET_PRACTICAL_BPW = 5.0
TARGET_STRETCH_BPW = 4.5
QUALITY_MIN = 0.95
QUALITY_AIM = 0.97

# H95 exponent complete-rate reference used only in packed estimates
# (physical exp bytes come from the adaptive codec at encode time).
H95_EXP_REF_BPW = 2.62

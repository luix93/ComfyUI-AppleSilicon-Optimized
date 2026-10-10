# Krea2 October 2026 research candidates

These are diagnostic prototypes, not runtime patches imported by this pack.
The accepted exact GEMM schedule is in `_patches/int8_ext/int8_gemm.mm`.
The INT8 attention candidates were **not promoted**: the extra savings were
small and the generated image bytes/content changed. See the benchmark report.

`attention_candidate.py` implements current CCV INT8 QK/PV attention, with
separate plain and centered-Hadamard modes, optional direct BHSD input/output,
and the rejected exact Q32 query-tile experiment. Generated Metal sources come
from liuliu/ccv `4b8d569` (October 8); its BSD license is retained here.

`tile_candidates.py` builds experimental INT8 GEMM tile schedules from the
loaded pack source. `final_candidate.py` toggles the accepted native schedule
against its previous route and combines it with the relaxed attention prototype.
The text-fusion cache in `text_candidate.py` also failed the complete-workflow
acceptance screen. Its simplified comparison/cache is diagnostic only.
They are sampler-stage hooks (`install()`, `set_mode()`) for the isolated
benchmark harness. Set `ASFP8_KREA2_GEMM_TILE=0` when reproducing the old GEMM
baseline. Compilation, captures and byte verification are not performance runs.
Capture mode writes model operands locally; never include those tensors in
benchmark publications. No model weights or private input media are included.

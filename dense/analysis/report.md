# Dense Analysis Report
## Dataflow Strategy Comparison: WS vs IS vs OS
### LLaMA 1-7B Workloads on NVIDIA T4 and AMD MI300x

---

## 1. Experimental Setup

### Hardware
| Hardware | Architecture | Memory | Bandwidth |
|---|---|---|---|
| NVIDIA T4 | Turing, Tensor Cores | 16GB GDDR6 | 300 GB/s |
| AMD MI300x | CDNA3, Matrix Cores | 192GB HBM3 | 5.3 TB/s |

### Workloads
13 matrix multiplication shapes derived from LLaMA 1-7B architecture:
- 4 attention projection layers (Q, K, V, Output) — shape (4096×4096×4096)
- 3 FFN layers (gate, up, down) — shapes up to (4096×11008×4096)
- 3 attention score computations (Q×Kᵀ) — seq lengths 512/1024/2048
- 3 attention output computations (scores×V) — seq lengths 512/1024/2048

### Dataflow Strategies
| Strategy | Stationary Operand | Key Tile Size |
|---|---|---|
| WS (Weight Stationary) | B (weights) | Large BLOCK_N=128 |
| IS (Input Stationary) | A (inputs) | Large BLOCK_M=128 |
| OS (Output Stationary) | C (outputs in registers) | Large BLOCK_M=BLOCK_N=128, BLOCK_K=64 |

---

## 2. Key Findings

### Finding 1 — Winning Dataflow is Hardware-Dependent

**T4: WS wins 11/13 shapes, IS wins 2/13**
- WS dominates all large shapes (projections, FFN)
- IS wins only on the two smallest attention shapes (M8, M11)
- T4's relatively limited memory bandwidth favors weight reuse (WS)

**MI300x: IS wins all 13/13 shapes**
- IS is the consistent winner regardless of shape or operation type
- MI300x's massive HBM3 bandwidth (5.3 TB/s) reduces the cost of
  streaming weights, making input reuse the optimal strategy

This is the most important finding: **the optimal dataflow strategy
is hardware-dependent, not workload-dependent.**

---

### Finding 2 — OS is Consistently the Worst Strategy

| Hardware | Mean OS Penalty | Max OS Penalty |
|---|---|---|
| T4 | 3.30x slower than best | 4.20x |
| MI300x | 1.26x slower than best | 1.63x |

OS performs worst on both hardware platforms. However the penalty
is dramatically higher on T4 (3.30x) than MI300x (1.26x). This is
because OS requires both A and B to stream through memory — on T4's
limited bandwidth this is very costly, while MI300x's HBM3 absorbs
the cost almost entirely.

---

### Finding 3 — Dataflow Sensitivity is Hardware-Dependent

| Hardware | Avg worst/best ratio (large) | Avg worst/best ratio (small) |
|---|---|---|
| T4 | 3.05x | 3.35x |
| MI300x | 1.25x | 1.25x |

On T4, dataflow choice has a **3x impact** on performance — choosing
the wrong dataflow is costly. On MI300x, dataflow choice barely
matters (1.25x spread) — all three strategies perform similarly due
to the abundant memory bandwidth.

**Practical implication:** Dataflow optimization is critical for
bandwidth-constrained hardware (T4-class) but less important for
bandwidth-abundant hardware (MI300x-class).

---

### Finding 4 — Small Attention Shapes Behave Differently on T4

On T4, IS edges out WS for the smallest attention shapes:
- M8_AttnScore_512: IS=47μs vs WS=48μs
- M11_AttnOut_512: IS=69μs vs WS=69μs

These shapes have a small K dimension (128), meaning weight tiles
are small and weight reuse is limited — reducing WS's advantage.
Input reuse (IS) becomes slightly more efficient at this scale.

On MI300x this distinction disappears — IS wins regardless of K size.

---

## 3. Figures

| Figure | Description |
|---|---|
| fig1_latency_by_dataflow.png | Latency per shape per dataflow, split large/small |
| fig2_winner_heatmap.png | Winning dataflow heatmap across hardware and shapes |
| fig3_os_penalty.png | OS penalty vs best dataflow per hardware |
| fig4_mi300x_speedup.png | MI300x speedup over T4 per dataflow |
| fig5_tflops_comparison.png | TFLOPS achieved per dataflow per hardware |

---

## 4. Conclusions

1. **Dataflow choice matters most on bandwidth-constrained hardware.**
   On T4, choosing WS over OS saves up to 4.2x in latency. On MI300x
   the same choice saves only 1.6x.

2. **IS is the optimal strategy for MI300x across all LLM workloads.**
   This holds consistently across all 13 shapes tested — projections,
   FFN layers, and attention computations at all sequence lengths.

3. **WS is the optimal strategy for T4 on large LLM shapes.**
   For the dominant compute operations in LLMs (large projections and
   FFN layers), WS consistently outperforms IS and OS on T4.

4. **OS should be avoided on bandwidth-constrained hardware.**
   The 3.3x average penalty on T4 makes OS a poor choice for any
   LLM inference or training workload on T4-class GPUs.





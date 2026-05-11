# Background: Dataflow Strategies

## Weight Stationary (WS)
Weights are preloaded into each PE and stay fixed. Input activations
stream through horizontally, partial sums move vertically.
- Minimizes weight movement
- Good when same weights are reused across many inputs
- Used by Google TPU MXU (fixed)

## Input Stationary (IS)
Input activations stay fixed in each PE. Weights stream through.
- Minimizes input movement
- Good when same inputs are reused across many weight matrices

## Output Stationary (OS)
Each PE accumulates a complete output element locally until fully
computed, then writes it out once.
- Minimizes partial sum movement
- Good when accumulation dimension K is large

## References
- Eyeriss: Chen et al., 2016
- SCALE-Sim: Samajdar et al., 2018

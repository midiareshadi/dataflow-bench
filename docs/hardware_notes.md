# Hardware Notes

## NVIDIA T4
- Architecture: Turing
- Tensor Cores: yes (FP16)
- Memory: 16GB GDDR6
- Memory bandwidth: 300 GB/s
- Available via: Google Colab free tier

## AMD MI300x
- Architecture: CDNA3
- Matrix Cores: yes (FP16/BF16)
- Memory: 192GB HBM3 (on-package)
- Memory bandwidth: 5.3 TB/s
- Available via: AMD Developer Cloud

## Google TPU v5e
- Systolic array (MXU): fixed WS dataflow
- Used in SCALE-Sim validation (PR submitted)
- Available via: Google Colab TPU runtime

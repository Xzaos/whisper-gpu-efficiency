# whisper-gpu-efficiency

What do mixed precision, gradient checkpointing, LoRA and 2-GPU data parallelism actually buy when fine-tuning whisper-small on a Kaggle "GPU T4 x2" machine? I measured each one with the same data, the same step count and a fixed sample order, and checked that the 2-GPU and gradient-accumulation setups compute the same thing as a plain big batch.

The numbers below are `_` until the Kaggle run has happened. Nothing here was measured yet; the code is tested on CPU with a tiny random Whisper config and smoke-tested on a small local GPU, not on the T4s.

## Setup

- Data: 400 FLEURS hi_in train clips (seed 0). Whisper log-mel features and label ids are computed once and saved to one tensor file, so a timed step is only the GPU.
- Each config runs 10 untimed warmup steps, then 40 timed steps. I record samples/s, ms/step, peak memory (`torch.cuda.max_memory_allocated`) and the loss at the last step. A CUDA OOM is caught, the cache is emptied and the cell says OOM.
- Full fine-tune uses AdamW at 1e-5, LoRA (r=32 on q, k, v, out_proj, fc1, fc2) uses 5e-4. Both clip gradients at 1.0 and use no schedule. Last-step losses are not comparable across configs, they only show training did not blow up.
- fp16 means autocast plus GradScaler, with the weights kept in fp32 (the T4 has no fast bf16). With LoRA the frozen base weights also stay fp32.
- The loss is a token sum divided by the token count of the whole global batch. I did that on purpose: averaging per-micro-batch means gives a different gradient when micro-batches hold different numbers of tokens, so batch 8 x accum 2 would not match batch 16.

## Results

Peak memory is in GiB.

| config | batch | samples/s | ms/step | peak GB | loss |
|---|---|---|---|---|---|
| 1 full fp32 | 8 | _ | _ | _ | _ |
| 2 full fp16 | 8 | _ | _ | _ | _ |
| 3 full fp16 + checkpointing | 8 | _ | _ | _ | _ |
| 4 LoRA fp16 | 8 | _ | _ | _ | _ |

Largest batch that fits, trying 4, 8, 16, 32, 64 and stopping at the first OOM (these sweep runs use 2 warmup and 4 timed steps, to keep the whole run under 25 minutes):

| config | max batch | samples/s at max | peak GB at max |
|---|---|---|---|
| 2 full fp16 | _ | _ | _ |
| 3 full fp16 + checkpointing | _ | _ | _ |
| 4 LoRA fp16 | _ | _ | _ |

DDP scaling, LoRA fp16, global batch 16:

| run | samples/s | ms/step | peak GB |
|---|---|---|---|
| 1 GPU, batch 16 (or 8 x accum 2 if 16 OOMs) | _ | _ | _ |
| 2 GPUs, 8 each | _ | _ | _ |

Scaling efficiency = throughput_2gpu / (2 x throughput_1gpu): _

Equivalence check: 30 steps of LoRA fp16 with dropout off and the same data order, as (a) one GPU batch 16, (b) one GPU batch 8 x accum 2, (c) DDP 2 x 8. I also rerun (b) to see how much two identical runs differ on their own.

| pair | max per-step abs loss diff |
|---|---|
| a vs b | _ |
| a vs c | _ |
| b vs c | _ |
| b vs b rerun | _ |

The full tables, GPU names and library versions are written to `results/report.md` and `results/efficiency.json` by the run.

## Running it

```bash
pip install -e ".[dev]"
pytest -q                      # CPU only, tiny random Whisper config, gloo for the DDP test
bash scripts/run_kaggle.sh     # on a Kaggle GPU T4 x2 machine, HF_TOKEN set
```

The script prints a timestamp per stage. Before each stage it runs `pip uninstall -y torchao` and sets `USE_TF=0` and `TRANSFORMERS_NO_TF=1`.

## Limitations

- 40 timed steps per config, one run each. I did not repeat runs, so small differences between configs are within noise I have not measured.
- One machine type (Kaggle T4 x2, PCIe, no NVLink), so the scaling number says nothing about other interconnects.
- whisper-small only.
- No FSDP or DeepSpeed, so nothing here says what sharding would buy.
- fp16 runs keep fp32 master weights under autocast; I did not try casting the frozen LoRA base to fp16.
- Features are copied from CPU to GPU every step, which is inside the timed region.

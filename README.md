# Whisper GPU Efficiency

What do mixed precision, gradient checkpointing, LoRA and 2-GPU data parallelism actually buy when fine-tuning whisper-small on a Kaggle "GPU T4 x2" machine? I measured each one with the same data, the same step count and a fixed sample order, and checked whether the 2-GPU and gradient-accumulation setups compute the same thing as a plain big batch.

Short version: fp16 is the big win for speed, checkpointing buys batch size and not speed, LoRA saved little here, and DDP scaled almost perfectly. The fp16 equivalence check did not pass, but the diagnosis shows the gap comes from nondeterministic GradScaler skips and not from DDP: in fp32, DDP matches batch 8 x accum 2 to 4.65e-04.

Everything below comes from `results/report.md` and `results/raw/*.json`, from one run on two Tesla T4s (torch 2.11.0+cu130, transformers 4.57.6, peft 0.21.2).

## Setup

- Data: 400 FLEURS hi_in train clips (seed 0). Whisper log-mel features and label ids are computed once and saved to one tensor file, so a timed step is only the GPU.
- Each config runs 10 untimed warmup steps, then 40 timed steps. I record samples/s, ms/step, peak memory (`torch.cuda.max_memory_allocated`, in GiB) and the loss at the last step. A CUDA OOM is caught, the cache is emptied and the cell says OOM.
- Full fine-tune uses AdamW at 1e-5, LoRA (r=32 on q, k, v, out_proj, fc1, fc2) uses 5e-4. Both clip gradients at 1.0 and use no schedule. Last-step losses are not comparable across configs, they only show training did not blow up.
- fp16 means autocast plus GradScaler, with the weights kept in fp32 (the T4 has no fast bf16). With LoRA the frozen base weights also stay fp32.
- The loss is a token sum divided by the token count of the whole global batch. Averaging per-micro-batch means gives a different gradient when micro-batches hold different numbers of tokens, so batch 8 x accum 2 would not match batch 16.

## Results

| config | batch | samples/s | ms/step | peak GB | loss |
|---|---|---|---|---|---|
| 1 full fp32 | 8 | 2.0 | 4043 | 12.79 | 0.560 |
| 2 full fp16 | 8 | 7.4 | 1075 | 10.21 | 0.569 |
| 3 full fp16 + checkpointing | 8 | 6.0 | 1325 | 4.76 | 0.568 |
| 4 LoRA fp16 | 8 | 7.0 | 1136 | 8.78 | 0.522 |

Largest batch that fits, trying 4, 8, 16, 32, 64 and stopping at the first OOM. These sweep runs used 2 warmup and 4 timed steps, to keep the whole run under 25 minutes.

| config | max batch | samples/s at max | peak GB at max |
|---|---|---|---|
| 2 full fp16 | 8 | 7.5 | 10.21 |
| 3 full fp16 + checkpointing | 32 | 6.4 | 9.54 |
| 4 LoRA fp16 | 8 | 7.0 | 8.78 |

### Mixed precision

fp16 autocast took throughput from 2.0 to 7.4 samples/s, 3.8 times faster. Peak memory went from 12.79 to 10.21 GiB, so the memory saving is much smaller than the speed gain: the weights, gradients and optimizer state stay in fp32.

### Gradient checkpointing

At batch 8 it cut peak memory from 10.21 to 4.76 GiB and cost 1075 to 1325 ms/step (23% slower). The largest batch went from 8 to 32. But throughput at batch 32 is 6.4 samples/s, still below plain fp16 at batch 8 (7.4). So checkpointing buys batch size and not speed. I did not test whether a larger batch helps final quality, which is the only reason I would want it here.

### LoRA

LoRA only dropped peak memory from 10.21 to 8.78 GiB and was not faster (7.0 vs 7.4 samples/s). It also did not raise the largest batch, which stays at 8. My reading is that activations dominate: Whisper pads every input to 30 s, and LoRA still backpropagates through the whole network, so it saves optimizer state and little else. I did not profile activations separately, so this is the explanation that fits the numbers and not something I measured.

This also explains why [whisper-hindi-telephony](https://github.com/Xzaos/whisper-hindi-telephony) ran out of memory at batch 16 on a T4 even with LoRA: here batch 16 is also OOM for LoRA fp16.

### DDP scaling

LoRA fp16, global batch 16. Batch 16 on one GPU ran out of memory, so the single-GPU baseline is 8 x accum 2.

| run | batch x accum x gpus = global | samples/s | ms/step | peak GB |
|---|---|---|---|---|
| 1 GPU | 8 x 2 x 1 = 16 | 7.1 | 2267 | 8.83 |
| 2 GPU DDP | 8 x 1 x 2 = 16 | 13.8 | 1159 | 8.82 |

Scaling efficiency = throughput_2gpu / (2 x throughput_1gpu) = 0.978. Only the LoRA gradients are all-reduced, which is small, so near-linear scaling is not surprising, but it is one machine and one run.

### Equivalence check

30 steps of LoRA fp16 with dropout off and the same data order. Run (a), one GPU at batch 16, ran out of memory, so there is no direct large-batch reference. I compared (b) one GPU at batch 8 x accum 2, a rerun of (b), and (c) DDP 2 x 8, then ran a diagnosis (`results/diagnose.md`) that logs inf gradients per rank, compares weight checksums across ranks and repeats b and c in fp32.

| comparison | max abs loss diff | at step |
|---|---|---|
| b vs c, fp16 | 4.85 | 9 |
| b vs b rerun, fp16 | 9.77 | 9 |
| b vs c, fp32 | 4.65e-04 | 18 |

The fp32 control agrees to 4.65e-04 at worst, so DDP, the loss normalisation and the sharding are correct. Both ranks together saw b's samples at every step. The replicas never diverge either: weight checksums were identical across ranks (max diff 0) in fp16 and in fp32. When one rank gets an inf, the all-reduce carries it to the other, so both skip the step together. At step 5 of the fp16 DDP run, rank 1 had a non-finite gradient in 1 parameter before the all-reduce and in 162 after it.

The cause is that fp16 GradScaler skip decisions are not deterministic across runs, and each skip forks the trajectory. In the diagnose run b overflowed at steps 5, 8 and 9 and c at steps 5 and 6. In the earlier run b skipped none. The parameters named in the overflow log are decoder layer 0 LoRA weights, early in training. A rerun of the identical b config diverged from b by 9.77, more than DDP did from accumulation (4.85), so the fp16 gap says nothing about DDP. The 1.1e-02 noise floor I reported earlier came from two runs that happened to make the same skip decisions.

I had guessed that scaling the loss by world size doubles the per-rank scaled gradients and makes DDP overflow more. That is rejected: b has no DDP and overflowed three times. The overflows also kept happening after the scale had backed off, at 32768 (c step 6, b step 8) and 16384 (b step 9), so halving the initial scale does not look like enough to avoid overflow in layer 0.

For equivalence checks I compare fp16 runs only with skip logging on, and use fp32, bf16 on hardware that supports it, or a much lower init scale.

On loss normalisation, which I checked in the code: the DDP path divides by the global token count, not each rank's local count. Every rank computes the token count from the full global batch slice (same seed, same order), so no all-reduce is needed, and the loss is scaled by world size so that DDP's gradient averaging gives the sum over the global batch. The fp32 control above confirms this on the GPUs.

## Running it

```bash
pip install -e ".[dev]"
pytest -q                      # CPU only, tiny random Whisper config, gloo for the DDP test
bash scripts/run_kaggle.sh     # on a Kaggle GPU T4 x2 machine, HF_TOKEN set
```

The script prints a timestamp per stage. Before each stage it runs `pip uninstall -y torchao` and sets `USE_TF=0` and `TRANSFORMERS_NO_TF=1`.

## Limitations

- 40 timed steps per config, and the max-batch sweep used only 4 timed steps per batch size.
- One machine type (Kaggle T4 x2, PCIe, no NVLink), so the scaling number says nothing about other interconnects.
- whisper-small only, one seed, one run per config.
- No FSDP or DeepSpeed, so nothing here says what sharding would buy.
- fp16 runs keep fp32 master weights under autocast; I did not try casting the frozen LoRA base to fp16.
- Features are copied from CPU to GPU every step, which is inside the timed region.
- The fp16 equivalence check cannot separate DDP from skip noise, so the DDP check rests on the fp32 run, which is one seed and 30 steps.

## Next steps

Rerun the fp16 equivalence with a much lower GradScaler `init_scale` and skip logging on, to see whether the runs stop skipping and agree.

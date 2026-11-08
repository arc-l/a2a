# A2A-GroundingModel

Fine-tuned grounding weights for Affordance2Action, implemented with
[SAM3-I](https://github.com/debby-0527/SAM3-I).

[Download from the Rutgers Box `model/` folder](https://rutgers.app.box.com/s/inc77v41aehwfdyp6utwmxpmhrwo1j2p/folder/435640657467).

The folder includes `SHA256SUMS` and `verify-downloads.ps1` for verifying all
model files. Run verification from the downloaded model folder.

| File | Purpose |
|---|---|
| `A2A-GroundingModel.pt` | Model tensors and adapter metadata; 7,292,676,741 bytes |
| `grounding_config.json` | Grounding export configuration |
| `model_manifest.json` | Export provenance, tensor validation, and checkpoint metadata |
| `MODELS_README.txt` | Model compatibility and dependency notes |

The checkpoint preserves 3,486 model entries and the adapter configuration, with
optimizer state removed. Every exported model tensor was compared exactly with
the source tensor on CPU. The checkpoint metadata records epoch 6 and `inst_stage=3`;
this export did not rerun model or robot evaluation.

SHA256 for `A2A-GroundingModel.pt`:

```text
3525a9163afe21b496add640614d05b1011e8cd5f5aff4d16a4b1a0fedc4936c
```

## Loading compatibility

`A2A-GroundingModel` is the release name; SAM3-I names the underlying implementation.
Its legacy `inst_stage=1_2` loader checks the checkpoint filename. For that route,
create a local alias and pass the alias to the upstream loader:

```bash
ln -s A2A-GroundingModel.pt sam3i_3_checkpoint.pt
```

The usual `inst_stage=1_1` inference route accepts the release filename directly.
The checkpoint, configuration, and notes are available now; the complete A2A
grounding training/inference integration is still being prepared. The weights
alone are not a standalone executable model package.

Training data are described in the [A2A-Bench training release](../A2A-Bench#download).
Upstream models and third-party data retain their original terms.

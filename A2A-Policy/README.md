# A2A-Policy: LIBERO simulation checkpoint

[Download from Rutgers Box](https://rutgers.box.com/s/inc77v41aehwfdyp6utwmxpmhrwo1j2p).

This release contains one **VLAUAD Arm A1 masked_tokens** policy for **LIBERO-Object**.
It is not the Piper real-robot Explicit policy or a reproduction package for the
paper's headline manipulation results.

| File | Purpose |
|---|---|
| `a2a_policy_libero_masked_tokens.pt` | Policy weights and export metadata; 223,367,181 bytes |
| `policy_args.json` | Architecture and policy configuration |
| `action_normalizer.json` | Action normalization values |
| `model_manifest.json` | Checkpoint variant and export validation |
| `A2A-GroundingModel.pt` | Matching online grounding checkpoint |

The export preserves 125 policy entries and the action normalizer, with optimizer
state removed. Tensors and normalization values were compared with the source
checkpoint. Internal machine paths were removed from the exported configuration.

The historical simulation evaluation recorded **440 successes in 500 episodes
(88%)**. This release did not rerun that evaluation and does not establish
real-robot deployability, real-time operation, or autonomous success detection.

SHA256 for `a2a_policy_libero_masked_tokens.pt`:

```text
f1cb5a37fa998a0cdeff5f66d927617da8e8c4c3be103fda18958fe3c4114d78
```

The policy also requires frozen DINOv2/MiniLM backbones and online SAM3-I grounding.
The complete policy training/inference integration and these external backbones
are not bundled in this release. See [A2A-GroundingModel](../A2A-GroundingModel)
and `MODELS_README.txt` for checkpoint compatibility notes.

Validation/test annotations and the real-robot policy remain pending. Upstream
models and datasets retain their original terms.

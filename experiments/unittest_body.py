# flake8: noqa -- executed after the COMMON prelude of build_runs.py (sys, os, json, log, pip, PIPELINE)
# CPU unit tests of the v3 components on SYNTHETIC flows (no dataset): first-contact hashing and counting,
# the causal context builder, class-balanced replay, context merging and a forward/backward pass of CITLGT.
pip("torch-geometric", "torchmetrics", "pydantic", "filelock", "psutil", "ruptures")
import importlib.util
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("rp", PIPELINE)
rp = importlib.util.module_from_spec(spec)
sys.modules["rp"] = rp  # dataclasses resolve string annotations through sys.modules
spec.loader.exec_module(rp)
device = torch.device("cpu")
cfg = rp.Config()
results = {}

# 1. pair_hash: int64 safe for negative/large inputs, deterministic, tag-separated, few collisions.
g = torch.Generator().manual_seed(0)
a = torch.randint(-2 ** 62, 2 ** 62, (200000,), generator=g)
b = torch.randint(-2 ** 62, 2 ** 62, (200000,), generator=g)
p = torch.randint(-1, 65536, (200000,), generator=g)
h1 = rp.pair_hash(a, b, p, torch.zeros_like(p), 1)
h2 = rp.pair_hash(a, b, p, torch.zeros_like(p), 2)
results["hash_dtype_int64"] = h1.dtype == torch.int64
results["hash_deterministic"] = bool(torch.equal(h1, rp.pair_hash(a, b, p, torch.zeros_like(p), 1)))
results["hash_tag_separates"] = bool((h1 != h2).all())
results["hash_unique_fraction"] = float(torch.unique(h1).numel() / h1.numel())


# 2. Synthetic stream: host A sweeps 300 destination ports of host B in 30 s; host C makes 300 flows to D:443.
def ip(v):
    return 0xFFFF00000000 | v


def batch_rows(start, count, scan, t0):
    t = t0 + torch.arange(count, dtype=torch.float64) * 0.05
    n = count
    rows = {"s0": torch.zeros(n, dtype=torch.long), "d0": torch.zeros(n, dtype=torch.long),
            "s1": torch.full((n,), ip(1 if scan else 3), dtype=torch.long),
            "d1": torch.full((n,), ip(2 if scan else 4), dtype=torch.long),
            "sport": torch.full((n,), 40000, dtype=torch.long),
            "dport": (torch.arange(start, start + count) + 1) if scan else torch.full((n,), 443, dtype=torch.long),
            "proto": torch.full((n,), 6, dtype=torch.long), "first": t, "last": t, "lo": t, "hi": t,
            "in_bytes": torch.full((n,), 60.0, dtype=torch.float64), "out_bytes": torch.zeros(n, dtype=torch.float64),
            "in_pkts": torch.ones(n, dtype=torch.float64), "out_pkts": torch.zeros(n, dtype=torch.float64)}
    return rows


state = rp.EndpointState(cfg, device, 0.0)
clock = 0.0
for start in range(0, 300, 25):
    for scan in (True, False):  # interleaved, strictly increasing availability times
        rows = batch_rows(start, 25, scan, clock)
        clock += 1.25
        state.advance(float(rows["hi"][0]))
        state.features(rows)
        state.observe(rows, torch.zeros(25))
state.advance(clock + 1.0)
keys = state.keys[:, 1].tolist()
stat = {name: i for i, name in enumerate(["flows", "packets", "bytes", "score", "newpeers", "newservices"])}


def total(host, name):
    k = keys.index(ip(host))
    return float(state.counts[k, 0, stat[name]])  # shortest (60 s) scale


results["scanner_new_services_60s"] = total(1, "newservices")
results["benign_client_new_services_60s"] = total(3, "newservices")
results["scanned_victim_new_services_60s"] = total(2, "newservices")
results["scanner_new_peers_60s"] = total(1, "newpeers")
results["first_contact_semantics_ok"] = (results["scanner_new_services_60s"] > 100 and results["benign_client_new_services_60s"] < 1.5
                                         and results["scanned_victim_new_services_60s"] > 100 and results["scanner_new_peers_60s"] < 1.5)
results["pair_table_size"] = int(len(state.pair_keys))

# 2b. v4 segment-relative context: finite, roughly standardised activity block, both intensities comparable.
cfg_norm = rp.Config(context_norm=True)
state_n = rp.EndpointState(cfg_norm, device, 0.0)
clock, blocks = 0.0, []
for start in range(0, 300, 25):
    for scan in (True, False):
        rows = batch_rows(start, 25, scan, clock)
        clock += 1.25
        state_n.advance(float(rows["hi"][0]))
        blocks.append(state_n.features(rows))
        state_n.observe(rows, torch.zeros(25))
feats = torch.cat(blocks)
activity = feats[:, :len(cfg_norm.memory_scales_s) * state_n.stats]
results["context_norm_finite"] = bool(torch.isfinite(feats).all())
results["context_norm_abs_mean"] = float(activity[len(activity) // 2:].mean(0).abs().mean())
results["context_norm_ok"] = results["context_norm_finite"] and results["context_norm_abs_mean"] < 2.0

# 3. Runtime + causal context + replay + merge + CITLGT forward/backward on synthetic model inputs.
F_NUM = 5
prep = {"classes": ["Benign", "scan"], "means": [0.0] * F_NUM, "scale": [1.0] * F_NUM, "vocabs": [["a", "b"]],
        "environments": ["e0"], "class_counts": [300, 300]}
manifest = {"max_interval_width": 0.0}
runtime = rp.Runtime(cfg, prep, manifest, device)
runtime.start_stream("s", "unix")
model = rp.CITLGT(prep, cfg, runtime.endpoints.width)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
replay = rp.ContextReplay(2, cfg, 0)
losses = []
clock = 0.0
for start in range(0, 300, 25):
    for scan in (True, False):
        rows = batch_rows(start, 25, scan, clock)
        clock += 1.25
        n = 25
        x = torch.cat((torch.randn(n, 2 * F_NUM), torch.full((n, 1), 2.0)), 1)  # numeric+missing, category code
        batch = {**rows, "x": x, "y": torch.full((n,), 1 if scan else 0, dtype=torch.long),
                 "env": torch.zeros(n, dtype=torch.long), "binary_y": torch.full((n,), 1 if scan else 0, dtype=torch.long),
                 "source_verified": torch.ones(n, dtype=torch.bool), "stream_key": torch.zeros(n, dtype=torch.long),
                 "relative_clock": torch.zeros(n, dtype=torch.bool), "source_order": torch.arange(start, start + n),
                 "flow_seq": torch.full((n,), -1, dtype=torch.long)}
        roots = runtime.prepare(batch)
        ctx = rp.local_context(runtime.flows, roots)
        items = replay.sample()
        merged = rp.merge_contexts([ctx] + [item[0] for item in items])
        y_all = torch.cat([batch["y"]] + [item[1] for item in items])
        logits, env_logits, z = model(merged)
        loss = torch.nn.functional.cross_entropy(logits, y_all)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss))
        replay.add(runtime.flows, roots, batch["y"], batch["env"])
        runtime.finish(batch, roots, torch.softmax(logits[:n].detach(), 1)[:, 1])
results["model_forward_backward"] = bool(np.isfinite(losses).all())
# 3b. v5 piecewise-linear encoding: exact encodings on known values and a finite forward/backward pass.
edges = [[-1.0, 0.0, 1.0, 1.0, 2.0]] * F_NUM  # includes one empty (zero-width) bin
prep_ple = {**prep, "feature_transform": {"version": 3, "log": [False] * F_NUM, "bit_columns": [], "mean": [0.0] * F_NUM,
                                          "scale": [1.0] * F_NUM, "ple_edges": edges}}
cfg_ple = rp.Config(numeric_embedding="ple", ple_bins=4)
model_ple = rp.CITLGT(prep_ple, cfg_ple, runtime.endpoints.width)
value = torch.tensor([[-2.0], [-0.5], [0.5], [1.0], [3.0]])[:, :, None]
ramp = (value - model_ple.ple_left[:1]) / model_ple.ple_width[:1].clamp_min(1e-6)
step = (value >= model_ple.ple_left[:1] + model_ple.ple_width[:1]).float()
enc = torch.where(model_ple.ple_width[:1] > 0, ramp.clamp(0, 1), step)[:, 0]
expected = torch.tensor([[0, 0, 0, 0], [0.5, 0, 0, 0], [1, 0.5, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.float32)
results["ple_encoding_exact"] = bool(torch.allclose(enc, expected))
out = model_ple(merged)[0]
out.float().sum().backward()
results["ple_forward_backward"] = bool(torch.isfinite(out).all()) and model_ple.input[0].weight.grad is not None
results["loss_first_last"] = [losses[0], losses[-1]]
results["replay"] = replay.statistics()
results["replay_mixed_classes"] = all(len(v) > 0 for v in replay.items)
results["context_width_matches_names"] = (len(rp.context_feature_names(cfg)) ==
                                          rp.compact_context(runtime, roots, cfg).shape[1])
print(json.dumps(results, indent=1))
required = ("hash_dtype_int64", "hash_deterministic", "hash_tag_separates", "first_contact_semantics_ok",
            "model_forward_backward", "replay_mixed_classes", "context_width_matches_names", "context_norm_ok",
            "ple_encoding_exact", "ple_forward_backward")
failed = [k for k in required if not results[k]]
if failed or results["hash_unique_fraction"] < 0.9999:
    raise SystemExit(f"unit tests failed: {failed}")
log("unit tests passed")

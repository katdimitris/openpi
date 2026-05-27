"""K-means initialization for concept vectors used by ConceptKDModule.

Collects teacher hidden tokens at the specified teacher layers (per modality) over a
representative subset of the training set, then runs k-means and writes the centers to
`assets/concept_kmeans/<config_name>/concepts.pt`.

Usage:
    uv run scripts/init_concept_kmeans.py <config_name> \
        --num-batches 50 \
        --output assets/concept_kmeans/<config_name>/concepts.pt

The output file is a single torch dict keyed by `<modality>_s<s>_t<t>`, matching the
ConceptKDModule expected layout. Set `concept_init_path` in the config to load it.
"""

import argparse
import dataclasses
import logging
import os
from pathlib import Path

import jax
import safetensors.torch
import torch
import torch.nn.functional as F  # noqa: N812
import tqdm

import openpi.models_pytorch.pi0_pytorch as _pi0_pytorch
import openpi.shared.normalize as _normalize  # noqa: F401
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@torch.no_grad()
def _kmeans_plus_plus_init(x: torch.Tensor, K: int, g: torch.Generator) -> torch.Tensor:
    """k-means++ seeding (Arthur & Vassilvitskii, 2007).

    Returns [K, D] seeds drawn from x with D^2 sampling. Distance updates are
    incremental so this is O(K*N*D), ~seconds on a GPU at our scales.
    """
    N, D = x.shape
    centers = torch.empty(K, D, device=x.device, dtype=x.dtype)
    idx = int(torch.randint(N, (1,), generator=g, device=x.device).item())
    centers[0] = x[idx]
    d2 = (x - centers[0]).pow(2).sum(dim=1)  # squared dist to nearest chosen
    for k in range(1, K):
        probs = d2 / (d2.sum() + 1e-12)
        idx = int(torch.multinomial(probs, 1, generator=g).item())
        centers[k] = x[idx]
        d2 = torch.minimum(d2, (x - centers[k]).pow(2).sum(dim=1))
    return centers


@torch.no_grad()
def kmeans(
    x: torch.Tensor,
    num_clusters: int,
    n_iter: int = 20,
    spherical: bool = False,
    seed: int = 0,
) -> torch.Tensor:
    """Vectorized k-means with k-means++ init and empty-cluster reseeding.

    Args:
      x: [N, D]. For spherical=True must be L2-normalized; centers are kept on the
        unit sphere after every update to match cosine similarity at training time.
      num_clusters: number of centers (K).
      n_iter: EM iterations.
      spherical: if True, run spherical k-means (argmax cosine instead of argmin L2).

    Returns:
      centers: [K, D]
    """
    g = torch.Generator(device=x.device).manual_seed(seed)
    N, D = x.shape
    if N < num_clusters:
        # Pad by repeating and adding tiny noise so duplicates aren't bit-identical.
        repeats = (num_clusters + N - 1) // N
        x_pad = x.repeat(repeats, 1)[:num_clusters]
        x_pad = x_pad + 1e-4 * torch.randn(num_clusters, D, generator=g, device=x.device, dtype=x.dtype)
        return F.normalize(x_pad, p=2, dim=1) if spherical else x_pad

    centers = _kmeans_plus_plus_init(x, num_clusters, g)
    if spherical:
        centers = F.normalize(centers, p=2, dim=1)

    ones = torch.ones(N, device=x.device, dtype=x.dtype)
    a_index = None  # set below for the empty-cluster reseed
    for _ in range(n_iter):
        if spherical:
            sim = x @ centers.t()                # [N, K]; both unit-norm
            coverage = sim.max(dim=1).values     # higher = better covered
            a = sim.argmax(dim=1)
        else:
            cc = (centers * centers).sum(dim=1)  # [K]
            xx = (x * x).sum(dim=1, keepdim=True)  # [N, 1]
            d2 = xx + cc.unsqueeze(0) - 2.0 * (x @ centers.t())
            coverage = -d2.min(dim=1).values     # higher = better covered
            a = d2.argmin(dim=1)
        a_index = a

        # Vectorized per-cluster mean via scatter_add.
        counts = torch.zeros(num_clusters, device=x.device, dtype=x.dtype).scatter_add_(0, a, ones)
        sums = torch.zeros(num_clusters, D, device=x.device, dtype=x.dtype).scatter_add_(
            0, a.unsqueeze(1).expand(N, D), x
        )
        empty = counts == 0
        new_centers = sums / counts.clamp(min=1).unsqueeze(1)

        # Reseed empty clusters to the points the current bank covers worst — robust
        # against degenerate k-means++ draws and lets every concept slot earn its keep.
        if empty.any():
            n_empty = int(empty.sum().item())
            worst = coverage.argsort()[:n_empty]
            new_centers[empty] = x[worst]

        if spherical:
            new_centers = F.normalize(new_centers, p=2, dim=1)
        centers = new_centers

    # Log final occupancy for sanity (cheap, runs once per bank).
    if a_index is not None:
        counts = torch.zeros(num_clusters, device=x.device, dtype=torch.long).scatter_add_(
            0, a_index, torch.ones(N, device=x.device, dtype=torch.long)
        )
        logging.info(
            f"  cluster occupancy: min={int(counts.min())} med={int(counts.median())} "
            f"max={int(counts.max())} empty_before_reseed={int(empty.sum()) if empty is not None else 0}"
        )
    return centers


def main():
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str)
    parser.add_argument("--num-batches", type=int, default=50)
    parser.add_argument("--max-tokens-per-modality", type=int, default=50_000)
    parser.add_argument("--kmeans-iters", type=int, default=20)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override config batch_size for the teacher forward. K-means runs on a "
        "single GPU (unlike DDP training), so the per-process batch is the full one — "
        "pass e.g. 32 on an 80GB H100 to avoid OOM in the SigLIP tower.",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = _config.get_config(args.config_name)
    if args.batch_size is not None:
        cfg = dataclasses.replace(cfg, batch_size=args.batch_size)
        logging.info(f"Overriding batch_size -> {args.batch_size} for k-means collection")
    model_cfg = cfg.model
    if not getattr(model_cfg, "use_concept_kd", False):
        raise ValueError(f"Config {args.config_name} does not enable concept KD")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build teacher.
    teacher_cfg = _config.get_config(model_cfg.teacher_config).model
    object.__setattr__(teacher_cfg, "dtype", cfg.pytorch_training_precision)
    teacher = _pi0_pytorch.PI0Pytorch(teacher_cfg).to(device).eval()
    weight_path = cfg.pytorch_weight_path_teacher or cfg.pytorch_weight_path
    safetensors.torch.load_model(teacher, os.path.join(weight_path, "model.safetensors"), strict=False)
    logging.info(f"Loaded teacher weights from {weight_path}")

    # Build data loader.
    loader = _data.create_data_loader(cfg, framework="pytorch", shuffle=True)

    teacher_layers = sorted({int(t) for _, t in model_cfg.concept_layer_pairs})
    modalities = tuple(model_cfg.concept_modalities)

    # Collect tokens.
    collected = {m: {t: [] for t in teacher_layers} for m in modalities}
    total = {m: {t: 0 for t in teacher_layers} for m in modalities}
    cap = args.max_tokens_per_modality
    action_horizon = model_cfg.action_horizon
    # Spread the cap evenly across batches. A single batch easily fills the visual cap
    # otherwise, which collapses scene diversity to one mini-batch of trajectories.
    per_batch_cap = max(1, cap // args.num_batches)

    teacher.paligemma_with_expert._capture_layer_set = set(teacher_layers)

    for i, (observation, actions) in enumerate(tqdm.tqdm(loader, total=args.num_batches, desc="collect")):
        if i >= args.num_batches:
            break
        observation = jax.tree.map(lambda x: x.to(device), observation)
        actions_d = actions.to(torch.float32).to(device)
        noise = teacher.sample_noise(actions_d.shape, device)
        time = teacher.sample_time(actions_d.shape[0], device)
        teacher.paligemma_with_expert._captured_hidden = {}
        with torch.no_grad():
            _ = teacher.eval_model(observation, actions_d, noise, time)
        captured = teacher.paligemma_with_expert._captured_hidden

        # Recompute shapes and the per-token validity mask. prefix_pad_masks[:, :num_visual]
        # is False for tokens from padded cameras (e.g. the zeroed right wrist on Libero),
        # so masking them out keeps k-means from spending centers on all-zero features.
        images, img_masks, lang_tokens, lang_masks, _state = teacher._preprocess_observation(observation, train=True)
        prefix_embs, prefix_pad_masks, _ = teacher.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        num_lang = lang_tokens.shape[1]
        num_visual = prefix_embs.shape[1] - num_lang
        visual_mask = prefix_pad_masks[:, :num_visual].bool()

        for t in teacher_layers:
            pre, suf = captured[t]
            for m in modalities:
                if total[m][t] >= cap:
                    continue
                if m == "visual":
                    tok = pre[:, :num_visual][visual_mask]
                elif m == "language":
                    sl = pre[:, num_visual : num_visual + num_lang]
                    tok = sl[lang_masks.bool()]
                else:  # action
                    tok = suf[:, -action_horizon:].reshape(-1, suf.shape[-1])
                # Per-batch subsample so the cap is filled across many batches, not just one.
                take = min(cap - total[m][t], per_batch_cap, tok.shape[0])
                if take <= 0:
                    continue
                if take < tok.shape[0]:
                    perm = torch.randperm(tok.shape[0], device=tok.device)[:take]
                    tok = tok[perm]
                collected[m][t].append(tok.detach().to(torch.float32).cpu())
                total[m][t] += tok.shape[0]

    teacher.paligemma_with_expert._capture_layer_set = None
    teacher.paligemma_with_expert._captured_hidden = {}

    # Run k-means per (modality, layer pair).
    num_concepts = {
        "visual": model_cfg.concept_num_visual,
        "language": model_cfg.concept_num_language,
        "action": model_cfg.concept_num_action,
    }
    # Match training: ConceptKDModule applies normalize_mean_std (per-feature, across the
    # sampled token dim) and uses cosine similarity. If we skip this, centers live in raw
    # teacher-hidden space and the cosine init is effectively random.
    similarity = getattr(model_cfg, "concept_similarity", "cosine")
    spherical = similarity == "cosine"
    out: dict[str, torch.Tensor] = {}
    for s, t in model_cfg.concept_layer_pairs:
        for m in modalities:
            key = f"{m}_s{s}_t{t}"
            tokens = torch.cat(collected[m][t], dim=0).to(device)
            # Per-feature standardize across all collected tokens.
            tokens = (tokens - tokens.mean(dim=0, keepdim=True)) / (
                tokens.std(dim=0, keepdim=True) + 1e-6
            )
            if spherical:
                tokens = F.normalize(tokens, p=2, dim=1)
            logging.info(
                f"k-means: {key} from {tokens.shape[0]} tokens -> {num_concepts[m]} centers "
                f"(spherical={spherical})"
            )
            centers = kmeans(
                tokens, num_concepts[m], n_iter=args.kmeans_iters, spherical=spherical
            )
            out[key] = centers.cpu()

    out_path = args.output or f"assets/concept_kmeans/{args.config_name}/concepts.pt"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    logging.info(f"Saved concept k-means init to {out_path}")


if __name__ == "__main__":
    main()

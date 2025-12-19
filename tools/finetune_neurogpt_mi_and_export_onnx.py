import argparse
import glob
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CsvSample:
    x_16x250: np.ndarray  # float32, shape (16, 250)
    y: int


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fine-tune NeuroGPT pretrained weights for 3-class MI and export an ONNX for QNMDsol."
    )
    p.add_argument("--neurogpt-root", default=r"D:\NeuroGPT", help="Path to cloned NeuroGPT repo root")
    p.add_argument(
        "--weights",
        default=os.path.join("model", "pytorch_model.bin"),
        help="Path to pretrained pytorch_model.bin (state_dict)",
    )
    p.add_argument(
        "--data-glob",
        default="training_data_*.csv",
        help="Glob for training CSVs (Timestamp, Ch0..Ch15)",
    )
    p.add_argument(
        "--label-map",
        default="left=0,right=1,attack=2",
        help="Comma list mapping filename-substring to class id, e.g. left=0,right=1,attack=2",
    )
    p.add_argument("--sample-rate", type=float, default=250.0, help="CSV sampling rate (Hz)")
    p.add_argument("--window-seconds", type=float, default=1.0, help="Window size in seconds (default 1.0)")
    p.add_argument("--stride-seconds", type=float, default=0.5, help="Stride in seconds (default 0.5)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze encoder/embedder/decoder transformer; train only decoding head.",
    )
    p.add_argument("--onnx-out", default=os.path.join("model", "neurogpt.onnx"))
    return p


def parse_label_map(s: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=", 1)
        out[k.strip().lower()] = int(v.strip())
    if not out:
        raise ValueError("Empty --label-map")
    return out


def infer_label_from_name(filename: str, label_map: Dict[str, int]) -> int | None:
    name = os.path.basename(filename).lower()
    for k, v in label_map.items():
        if k in name:
            return v
    return None


def load_csv_windows(
    csv_path: str,
    label: int,
    sample_rate_hz: float,
    window_seconds: float,
    stride_seconds: float,
) -> List[CsvSample]:
    df = pd.read_csv(csv_path)
    # (n_samples, 16)
    data = df.iloc[:, 1:17].to_numpy(dtype=np.float32, copy=True)
    # µV -> V (matches their repo expectations; we normalize later anyway)
    data *= 1e-6
    # transpose to (16, n_samples)
    data = data.T
    n_channels, n_samples = data.shape
    if n_channels != 16:
        raise ValueError(f"Expected 16 channels, got {n_channels} in {csv_path}")

    win = int(round(sample_rate_hz * window_seconds))
    stride = int(round(sample_rate_hz * stride_seconds))
    if win <= 0 or stride <= 0:
        raise ValueError("Invalid window/stride")
    if n_samples < win:
        return []

    out: List[CsvSample] = []
    for start in range(0, n_samples - win + 1, stride):
        seg = data[:, start : start + win]  # (16, win)
        # QNMDsol runtime uses 250 timesteps; enforce/convert here.
        seg_250 = resample_linear(seg, sample_rate_hz, 250.0, target_len=250)
        out.append(CsvSample(x_16x250=seg_250, y=label))
    return out


def resample_linear(x: np.ndarray, src_hz: float, dst_hz: float, target_len: int) -> np.ndarray:
    # x: (C, N)
    c, n = x.shape
    if n == target_len and abs(src_hz - dst_hz) < 1e-6:
        return x.astype(np.float32, copy=False)
    duration = n / max(src_hz, 1e-6)
    t_dst = np.linspace(0.0, duration, num=target_len, endpoint=False, dtype=np.float32)
    t_src = (np.arange(n, dtype=np.float32) / max(src_hz, 1e-6)).astype(np.float32)
    y = np.empty((c, target_len), dtype=np.float32)
    for ch in range(c):
        y[ch] = np.interp(t_dst, t_src, x[ch].astype(np.float32, copy=False)).astype(np.float32)
    return y


class QnmdNeuroGptWrapper(nn.Module):
    """
    ONNX-export friendly wrapper:
    input: (B, 16, 250) float32
    internal:
      - pad channels to 22 (zeros)
      - upsample time 250->500 (linear)
      - build NeuroGPT batch dict with chunks=1
      - output logits: (B, num_classes)
    """

    def __init__(self, neuro_model: nn.Module, num_classes: int = 3):
        super().__init__()
        self.neuro_model = neuro_model
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,16,250)
        if x.ndim != 3:
            raise RuntimeError(f"Expected (B,16,250), got {tuple(x.shape)}")
        b, c, t = x.shape
        if c != 16:
            raise RuntimeError(f"Expected 16 channels, got {c}")

        # normalize per-window (simple)
        x = x - x.mean(dim=-1, keepdim=True)
        x = x / (x.std(dim=-1, keepdim=True) + 1e-6)

        # pad to 22 channels
        pad_ch = 22 - 16
        if pad_ch > 0:
            zeros = torch.zeros((b, pad_ch, t), dtype=x.dtype, device=x.device)
            x = torch.cat([x, zeros], dim=1)  # (B,22,250)

        # upsample time to 500
        x = F.interpolate(x, size=500, mode="linear", align_corners=False)  # (B,22,500)

        # add chunks dim: (B,1,22,500)
        x = x.unsqueeze(1)
        attn = torch.ones((b, 1), dtype=torch.long, device=x.device)
        outputs = self.neuro_model({"inputs": x, "attention_mask": attn})
        if isinstance(outputs, dict) and "decoding_logits" in outputs:
            return outputs["decoding_logits"]
        if isinstance(outputs, dict) and "outputs" in outputs:
            # Fallback: if not in decoding mode, return pooled features.
            # This should not happen in normal fine-tune.
            pooled = outputs["outputs"][:, -1, :]
            return pooled
        raise RuntimeError("Unexpected NeuroGPT output format")


def main() -> int:
    args = build_argparser().parse_args()
    label_map = parse_label_map(args.label_map)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Import NeuroGPT repo modules.
    neuro_root = os.path.abspath(args.neurogpt_root)
    sys.path.insert(0, os.path.join(neuro_root, "src"))
    from encoder.conformer_braindecode import EEGConformer  # type: ignore
    from decoder.make_decoder import make_decoder  # type: ignore
    from decoder.unembedder import make_unembedder  # type: ignore
    from embedder.make import make_embedder  # type: ignore
    from model import Model  # type: ignore

    weights_path = os.path.abspath(args.weights)
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    csv_files = sorted(glob.glob(args.data_glob))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files matched: {args.data_glob}")

    samples: List[CsvSample] = []
    for f in csv_files:
        y = infer_label_from_name(f, label_map)
        if y is None:
            continue
        samples.extend(
            load_csv_windows(
                f,
                y,
                sample_rate_hz=float(args.sample_rate),
                window_seconds=float(args.window_seconds),
                stride_seconds=float(args.stride_seconds),
            )
        )
    if not samples:
        raise RuntimeError("No labeled windows found; check --label-map and --data-glob")

    xs = np.stack([s.x_16x250 for s in samples], axis=0)  # (N,16,250)
    ys = np.array([s.y for s in samples], dtype=np.int64)

    num_classes = len(set(int(v) for v in ys.tolist()))
    if num_classes != 3:
        print(f"[warn] detected classes={sorted(set(ys.tolist()))}, but you asked for 3-class mapping.")
        # Still force to max(label)+1 for head shape.
        num_classes = int(ys.max()) + 1

    device = torch.device("cpu")

    # Build a "small" config that is compatible with the pretrained state_dict keys.
    # We keep the encoder at (22, 500) and use chunks=1.
    model_config = {
        "training_style": "CSM_causal",
        "architecture": "GPT",
        "pretrained_model": weights_path,
        "use_encoder": True,
        "ft_only_encoder": False,
        "embedding_dim": 1024,
        "num_hidden_layers_embedding_model": 1,
        "num_hidden_layers_unembedding_model": 1,
        "num_hidden_layers": 6,
        "num_attention_heads": 16,
        "intermediate_dim_factor": 4,
        "hidden_activation": "gelu_new",
        "dropout": 0.1,
        "n_positions": 512,
        "chunk_len": 500,
        "num_chunks": 1,
        "num_decoding_classes": num_classes,
        "filter_time_length": 25,
        "pool_time_length": 75,
        "stride_avg_pool": 15,
        "n_filters_time": 40,
        "num_encoder_layers": 6,
        "device": "cpu",
    }

    encoder = EEGConformer(
        n_outputs=model_config["num_decoding_classes"],
        n_chans=22,
        n_times=model_config["chunk_len"],
        ch_pos=None,
        is_decoding_mode=model_config["ft_only_encoder"],
        n_filters_time=model_config["n_filters_time"],
        filter_time_length=model_config["filter_time_length"],
        pool_time_length=model_config["pool_time_length"],
        pool_time_stride=model_config["stride_avg_pool"],
        att_depth=model_config["num_encoder_layers"],
    )
    model_config["parcellation_dim"] = (
        (
            (model_config["chunk_len"] - model_config["filter_time_length"] + 1 - model_config["pool_time_length"])
            // model_config["stride_avg_pool"]
            + 1
        )
        * model_config["n_filters_time"]
    )

    embedder = make_embedder(
        training_style=model_config["training_style"],
        architecture=model_config["architecture"],
        in_dim=model_config["parcellation_dim"],
        embed_dim=model_config["embedding_dim"],
        num_hidden_layers=model_config["num_hidden_layers_embedding_model"],
        dropout=model_config["dropout"],
        n_positions=model_config["n_positions"],
    )
    decoder = make_decoder(
        architecture=model_config["architecture"],
        num_hidden_layers=model_config["num_hidden_layers"],
        embed_dim=model_config["embedding_dim"],
        num_attention_heads=model_config["num_attention_heads"],
        n_positions=model_config["n_positions"],
        intermediate_dim_factor=model_config["intermediate_dim_factor"],
        hidden_activation=model_config["hidden_activation"],
        dropout=model_config["dropout"],
    )
    unembedder = make_unembedder(
        embed_dim=model_config["embedding_dim"],
        num_hidden_layers=model_config["num_hidden_layers_unembedding_model"],
        out_dim=model_config["parcellation_dim"],
        dropout=model_config["dropout"],
    )

    model = Model(encoder=encoder, embedder=embedder, decoder=decoder, unembedder=unembedder).to(device)
    model.from_pretrained(weights_path)
    model.switch_decoding_mode(is_decoding_mode=True, num_decoding_classes=num_classes)
    model.eval()

    if args.freeze_backbone:
        for name, p in model.named_parameters():
            p.requires_grad = "decoding_head" in name

    wrapper = QnmdNeuroGptWrapper(model, num_classes=num_classes).to(device)

    ds_x = torch.from_numpy(xs)
    ds_y = torch.from_numpy(ys)

    opt = torch.optim.AdamW([p for p in wrapper.parameters() if p.requires_grad], lr=float(args.lr))
    loss_fn = nn.CrossEntropyLoss()

    wrapper.train()
    n = ds_x.shape[0]
    bs = int(args.batch_size)
    for epoch in range(int(args.epochs)):
        perm = torch.randperm(n)
        total_loss = 0.0
        correct = 0
        for i in range(0, n, bs):
            idx = perm[i : i + bs]
            xb = ds_x[idx].to(device)
            yb = ds_y[idx].to(device)
            opt.zero_grad(set_to_none=True)
            logits = wrapper(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * int(xb.shape[0])
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
        print(f"epoch {epoch+1}/{args.epochs} loss={total_loss/n:.4f} acc={correct/n:.3f}")

    wrapper.eval()

    onnx_out = os.path.abspath(args.onnx_out)
    os.makedirs(os.path.dirname(onnx_out) or ".", exist_ok=True)
    dummy = torch.randn(1, 16, 250, dtype=torch.float32, device=device)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy,
            onnx_out,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    print(f"[ok] exported ONNX: {onnx_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

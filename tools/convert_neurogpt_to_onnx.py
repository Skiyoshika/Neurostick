import argparse
import importlib.util
import os
from collections import OrderedDict

import torch


def _import_module_from_path(module_path: str):
    module_path = os.path.abspath(module_path)
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Model definition not found: {module_path}")
    spec = importlib.util.spec_from_file_location("model_source", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import module from: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_state_dict(obj):
    if isinstance(obj, OrderedDict):
        return obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in obj and isinstance(obj[key], (dict, OrderedDict)):
                return obj[key]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    raise TypeError(f"Unrecognized checkpoint format: {type(obj)}")


def _remap_prefixes(state_dict, prefixes_to_strip):
    remapped = OrderedDict()
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes_to_strip:
            if new_k.startswith(p):
                new_k = new_k[len(p) :]
        remapped[new_k] = v
    return remapped


def _try_load(model, state_dict):
    try:
        result = model.load_state_dict(state_dict, strict=True)
        return True, result
    except RuntimeError:
        result = model.load_state_dict(state_dict, strict=False)
        return False, result


def load_state_dict_flexible(model, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)

    candidates = []
    candidates.append(state_dict)

    prefixes = ("module.", "model.", "net.", "backbone.", "encoder.", "neurogpt.")
    candidates.append(_remap_prefixes(state_dict, prefixes))

    # Also try stripping only the most common DP prefix first.
    candidates.append(_remap_prefixes(state_dict, ("module.",)))

    last_result = None
    for cand in candidates:
        strict_loaded, result = _try_load(model, cand)
        last_result = (strict_loaded, result, cand)
        if strict_loaded:
            return strict_loaded, result

        # If strict=False has no missing/unexpected keys, accept it.
        missing = list(getattr(result, "missing_keys", []))
        unexpected = list(getattr(result, "unexpected_keys", []))
        if not missing and not unexpected:
            return strict_loaded, result

    strict_loaded, result, _ = last_result
    return strict_loaded, result


def build_argparser():
    p = argparse.ArgumentParser(description="Convert NeuroGPT PyTorch weights to ONNX.")
    p.add_argument("--model-def", default="model_source.py", help="Path to model_source.py")
    p.add_argument("--model-class", default="NeuroGPT", help="Model class name inside model_source.py")
    p.add_argument("--weights", default=os.path.join("model", "pytorch_model.bin"), help="Path to .bin weights")
    p.add_argument("--onnx-out", default=os.path.join("models", "neurogpt.onnx"), help="Output ONNX path")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--input-shape", default="1,16,250", help="Dummy input shape: B,C,T")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Export device")
    return p


def main():
    args = build_argparser().parse_args()

    module = _import_module_from_path(args.model_def)
    if not hasattr(module, args.model_class):
        raise AttributeError(
            f"Class {args.model_class!r} not found in {os.path.abspath(args.model_def)}"
        )

    model_cls = getattr(module, args.model_class)

    try:
        model = model_cls()
    except TypeError as e:
        raise TypeError(
            f"Failed to instantiate {args.model_class} with no args. "
            f"Pass constructor args by editing this script or wrap your model class. Details: {e}"
        )

    model.eval()

    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Weights not found: {os.path.abspath(args.weights)}")

    strict_loaded, load_result = load_state_dict_flexible(model, args.weights)
    missing = list(getattr(load_result, "missing_keys", []))
    unexpected = list(getattr(load_result, "unexpected_keys", []))
    if not strict_loaded:
        print("[warn] load_state_dict(strict=False) was used due to key mismatch.")
    if missing:
        print(f"[warn] missing_keys ({len(missing)}): {missing[:50]}")
    if unexpected:
        print(f"[warn] unexpected_keys ({len(unexpected)}): {unexpected[:50]}")

    device = torch.device(args.device if args.device == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    b, c, t = (int(x.strip()) for x in args.input_shape.split(","))
    dummy_input = torch.randn(b, c, t, device=device)

    onnx_out = os.path.abspath(args.onnx_out)
    os.makedirs(os.path.dirname(onnx_out) or ".", exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            onnx_out,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f"[ok] Exported ONNX: {onnx_out}")


if __name__ == "__main__":
    main()

"""Quick environment check: CUDA visibility + headless OpenGL via EGL."""

import sys


def main() -> int:
    import torch

    print(f"python:  {sys.version.split()[0]}")
    print(f"torch:   {torch.__version__}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        cap = torch.cuda.get_device_capability(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
        print(f"cuda:    {torch.cuda.get_device_name(idx)} (sm_{cap[0]}{cap[1]}) | {vram:.1f} GB")
    elif torch.backends.mps.is_available():
        print("cuda:    not available; using MPS")
    else:
        print("cuda:    not available; using CPU")

    import moderngl

    try:
        ctx = moderngl.create_context(standalone=True, backend="egl")
        print(f"gl egl:  {ctx.info['GL_VERSION']}")
        print(f"renderer:{ctx.info['GL_RENDERER']}")
    except Exception as exc:
        print(f"gl egl:  FAILED — {exc!r}")
        print("         try: sudo apt install libegl1 libgl1 libegl-mesa0")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

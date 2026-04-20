from pathlib import Path
from typing import Annotated

import typer

from gleam.config import DEFAULT_DATASET_DIR, DEFAULT_OUTPUT_DIR, DEFAULT_SEED

app = typer.Typer(
    name="gleam",
    help="Neural Phong shader (cGAN) with ModernGL renderer.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


@app.command("generate-data")
def generate_data(
    out: Annotated[Path, typer.Option(help="Output dataset directory.")] = DEFAULT_DATASET_DIR,
    n: Annotated[int, typer.Option(help="Number of samples to render.")] = 3000,
    seed: Annotated[int, typer.Option(help="RNG seed.")] = DEFAULT_SEED,
) -> None:
    """Render `n` Phong images with randomized parameters."""
    from gleam.data.generate import generate_dataset
    from gleam.utils.logging import setup_logging

    setup_logging(DEFAULT_OUTPUT_DIR / "logs", run_name="generate")
    generate_dataset(out_dir=out, num_samples=n, seed=seed)


@app.command("train")
def train(
    dataset: Annotated[Path, typer.Option(help="Dataset directory.")] = DEFAULT_DATASET_DIR,
    output: Annotated[Path, typer.Option(help="Checkpoints / logs directory.")] = DEFAULT_OUTPUT_DIR,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 300,
    batch_size: Annotated[int, typer.Option(help="Batch size.")] = 32,
    resume: Annotated[Path | None, typer.Option(help="Resume from checkpoint.")] = None,
    device: Annotated[
        str | None, typer.Option(help="Force device: cuda | mps | cpu (auto if unset).")
    ] = None,
    num_workers: Annotated[
        int | None, typer.Option(help="DataLoader workers (auto if unset).")
    ] = None,
    amp: Annotated[bool, typer.Option(help="Enable BF16 autocast (CUDA/CPU only).")] = False,
    compile_: Annotated[
        bool, typer.Option("--compile/--no-compile", help="Enable torch.compile (CUDA only).")
    ] = False,
) -> None:
    """Train the cGAN generator on the Phong dataset."""
    from gleam.training.trainer import run_training
    from gleam.utils.logging import setup_logging

    setup_logging(output / "logs", run_name="train")
    run_training(
        dataset_dir=dataset,
        output_dir=output,
        epochs=epochs,
        batch_size=batch_size,
        resume=resume,
        device_override=device,
        num_workers=num_workers,
        use_amp=amp,
        use_compile=compile_,
    )


@app.command("eval")
def evaluate(
    ckpt: Annotated[Path, typer.Option(help="Path to generator checkpoint.")],
    dataset: Annotated[Path, typer.Option(help="Dataset directory.")] = DEFAULT_DATASET_DIR,
    output: Annotated[Path, typer.Option(help="Metrics output dir.")] = DEFAULT_OUTPUT_DIR / "eval",
    split: Annotated[str, typer.Option(help="Split: train | val | test.")] = "test",
    device: Annotated[
        str | None, typer.Option(help="Force device: cuda | mps | cpu (auto if unset).")
    ] = None,
    ema: Annotated[
        bool,
        typer.Option(
            "--ema/--raw",
            help="Use the EMA generator (default) or the raw training generator.",
        ),
    ] = True,
) -> None:
    """Evaluate neural renderer vs. ground truth on the chosen split."""
    from gleam.eval.runner import run_evaluation
    from gleam.utils.logging import setup_logging

    setup_logging(output, run_name="eval")
    run_evaluation(
        ckpt=ckpt,
        dataset_dir=dataset,
        output_dir=output,
        split=split,
        device=device,
        use_ema=ema,
    )


@app.command("report")
def report(
    ckpt: Annotated[
        Path, typer.Option(help="Path to generator checkpoint.")
    ] = DEFAULT_OUTPUT_DIR / "checkpoints" / "ema_generator.pt",
    dataset: Annotated[Path, typer.Option(help="Dataset directory.")] = DEFAULT_DATASET_DIR,
    output: Annotated[
        Path, typer.Option(help="Report output dir.")
    ] = DEFAULT_OUTPUT_DIR / "report",
    split: Annotated[str, typer.Option(help="Split: train | val | test.")] = "test",
    n_best: Annotated[int, typer.Option(help="Samples in visual_best grid.")] = 8,
    n_worst: Annotated[int, typer.Option(help="Samples in visual_worst grid.")] = 8,
    n_random: Annotated[int, typer.Option(help="Samples in visual_random grid.")] = 8,
    ema: Annotated[
        bool,
        typer.Option(
            "--ema/--raw",
            help="Use the EMA generator (default) or the raw training generator.",
        ),
    ] = True,
    device: Annotated[
        str | None, typer.Option(help="Force device: cuda | mps | cpu (auto if unset).")
    ] = None,
    seed: Annotated[int, typer.Option(help="RNG seed for the random sample picker.")] = 0,
) -> None:
    """Generate every artefact needed for the project report (tables + visuals)."""
    from gleam.eval.report import build_report
    from gleam.utils.logging import setup_logging

    setup_logging(output, run_name="report")
    build_report(
        ckpt=ckpt,
        dataset_dir=dataset,
        output_dir=output,
        split=split,
        n_best=n_best,
        n_worst=n_worst,
        n_random=n_random,
        use_ema=ema,
        device=device,
        seed=seed,
    )


@app.command("render")
def render_single(
    ckpt: Annotated[Path, typer.Option(help="Generator checkpoint.")],
    object_pos: Annotated[tuple[float, float, float], typer.Option()] = (5.0, 0.0, 0.0),
    light_pos: Annotated[tuple[float, float, float], typer.Option()] = (15.0, 5.0, 0.0),
    kd: Annotated[tuple[int, int, int], typer.Option(help="Diffuse color 0-255.")] = (200, 50, 50),
    shininess: Annotated[float, typer.Option()] = 10.0,
    out: Annotated[Path, typer.Option()] = Path("render.png"),
    compare: Annotated[bool, typer.Option(help="Also render GT side-by-side.")] = False,
) -> None:
    """Render a single image with the neural renderer (optionally side-by-side with GT)."""
    from gleam.eval.render_single import render_single_image
    from gleam.utils.logging import setup_logging

    setup_logging(run_name="render")
    render_single_image(
        ckpt=ckpt,
        object_pos=object_pos,
        light_pos=light_pos,
        kd=kd,
        shininess=shininess,
        out_path=out,
        compare_with_gt=compare,
    )


if __name__ == "__main__":
    app()

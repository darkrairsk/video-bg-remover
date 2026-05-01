"""
bgremover/cli.py
----------------
Argparse definitions and main() entry point for bgremover v2.
"""

import shutil
import sys
from pathlib import Path

from rich.console import Console

from bgremover import __version__
from bgremover import config as cfg_module
from bgremover import background, models, pipeline
from bgremover.encoder import resolve_output_path

console = Console()

_VIDEOS_DIR = Path(__file__).resolve().parent.parent / "videos"


def build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="remove_bg",
        description=(
            f"bgremover v{__version__} — Video background removal "
            "using Robust Video Matting → ProRes 4444 with alpha."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python remove_bg.py --input videos/selfie.mov
  python remove_bg.py --input videos/selfie.mov --bg-image assets/bg_image.jpg
  python remove_bg.py --input videos/selfie.mov --format webm
  python remove_bg.py --input videos/selfie.mov --preview 5
  python remove_bg.py --input videos/selfie.mov --save-config
  python remove_bg.py --show-config
        """,
    )

    # ── Core ──────────────────────────────────────────────────────────────
    core = parser.add_argument_group("Core")
    core.add_argument(
        "--input", metavar="PATH",
        help="Input video (.mov / .mp4). Required unless --show-config is used.",
    )
    core.add_argument(
        "--output", metavar="PATH", default=None,
        help=(
            "Output path. Defaults to "
            "videos/<input_stem>_processed.<ext>"
        ),
    )
    core.add_argument(
        "--bg-image", metavar="PATH", dest="bg_image", default=None,
        help=(
            "Clean background image. Optional for RVM; "
            "required for --model-type bgmv2. "
            "If omitted, checks assets/ then auto-extracts from frames."
        ),
    )

    # ── Model ─────────────────────────────────────────────────────────────
    mdl = parser.add_argument_group("Model")
    mdl.add_argument(
        "--model-type", dest="model_type",
        choices=["rvm", "rvm_resnet50", "bgmv2"], default=None,
        help="rvm (default) | rvm_resnet50 (slower, better edges) | bgmv2",
    )
    mdl.add_argument(
        "--model", metavar="PATH", default=None,
        help="Path to a custom TorchScript model file.",
    )
    mdl.add_argument(
        "--device", choices=["auto", "mps", "cuda", "cpu"], default=None,
        help="Compute device (default: auto → mps → cuda → cpu).",
    )
    mdl.add_argument(
        "--internal-res", dest="internal_res", type=int, default=None,
        help="Target short-side resolution for RVM inference (default: 512).",
    )

    # ── Output ────────────────────────────────────────────────────────────
    out = parser.add_argument_group("Output")
    out.add_argument(
        "--format", dest="format", choices=["prores", "webm", "png"], default=None,
        help="Output format: prores (default, CapCut-ready), webm (web/smaller), png (frame sequence).",
    )
    out.add_argument(
        "--guided-filter", action="store_true", default=None,
        help="Apply OpenCV Guided Filter to smooth AI mask against original frame.",
    )
    out.add_argument(
        "--no-guided-filter", action="store_false", dest="guided_filter", default=None,
        help="Disable Guided Filter.",
    )

    # ── Workflow ──────────────────────────────────────────────────────────
    wf = parser.add_argument_group("Workflow")
    wf.add_argument(
        "--preview", metavar="N", type=int, default=None,
        help="Process N evenly-spaced frames, save as PNGs, and exit. Great for a quick sanity check.",
    )
    wf.add_argument(
        "--save-config", action="store_true",
        help="Save this run's arguments as defaults in config.json.",
    )
    wf.add_argument(
        "--show-config", action="store_true",
        help="Print the current config.json settings and exit.",
    )
    wf.add_argument(
        "--version", action="version", version=f"bgremover {__version__}",
    )

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── --show-config (no --input needed) ──────────────────────────────
    if args.show_config:
        print(cfg_module.show())
        sys.exit(0)

    # ── Load config ─────────────────────────────────────────────────────
    cfg = cfg_module.load()

    # ── Interactive Mode if no --input ──────────────────────────────────
    if not args.input:
        from rich.prompt import Prompt, Confirm
        try:
            console.print("\n[bold cyan]✨ bgremover Interactive Menu ✨[/bold cyan]")
            console.print("[dim]Press Ctrl+C to exit at any time.[/dim]\n")
            
            while True:
                video_path = Prompt.ask("[bold]🎬 Enter input video path[/bold]")
                p = Path(video_path).expanduser().resolve()
                if p.exists() and p.is_file():
                    args.input = str(p)
                    break
                else:
                    console.print(f"[red]❌ File not found: {video_path}[/red]")
                    
            args.format = Prompt.ask(
                "[bold]🗂️  Select output format[/bold]",
                choices=["prores", "webm", "png"],
                default=cfg.get("format", "prores")
            )
            
            args.model_type = Prompt.ask(
                "[bold]🧠 Select AI model[/bold]",
                choices=["rvm", "rvm_resnet50", "bgmv2"],
                default=cfg.get("model_type", "rvm")
            )
            
            args.internal_res = int(Prompt.ask(
                "[bold]📐 Internal Resolution (higher = sharper edges, slower)[/bold]",
                default=str(cfg.get("internal_res", 512))
            ))
            
            args.guided_filter = Confirm.ask(
                "[bold]🧹 Enable Guided Filter post-processing for smoother edges?[/bold]",
                default=cfg.get("guided_filter", False)
            )
            
            if Confirm.ask("[bold]🔍 Run a quick 5-frame preview first?[/bold]", default=False):
                args.preview = 5
                
            console.print("\n[bold green]Ready![/bold green] Starting pipeline...\n")
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled by user.[/yellow]")
            sys.exit(0)

    # ── Merge config with args ──────────────────────────────────────────
    cfg = cfg_module.merge_args(cfg, args)

    model_type: str = cfg["model_type"]
    fmt:        str = cfg["format"]
    max_short:  int = cfg["max_short_side"]
    device_pref:str = cfg["device"]
    internal_res:int= cfg["internal_res"]
    guided_filter:bool=cfg["guided_filter"]

    # ── Validate input ──────────────────────────────────────────────────
    input_path = Path(args.input).resolve()
    if not input_path.exists():
        sys.exit(f"❌  Input video not found: {input_path}")

    # ── Ensure videos/ dir exists and copy raw video in ─────────────────
    _VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    raw_dest = _VIDEOS_DIR / input_path.name
    if input_path.parent.resolve() != _VIDEOS_DIR.resolve():
        if not raw_dest.exists():
            console.print(f"📂  Copying raw video into videos/ …")
            shutil.copy2(str(input_path), str(raw_dest))
        else:
            console.print(f"📂  Raw video already in videos/")

    # ── Resolve output path ──────────────────────────────────────────────
    output_path = resolve_output_path(
        input_path, args.output, fmt, cfg["output_suffix"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"📁  Output → [underline]{output_path}[/underline]")

    # ── --save-config ────────────────────────────────────────────────────
    if args.save_config:
        to_save = {}
        if args.model_type:
            to_save["model_type"] = args.model_type
        if args.format:
            to_save["format"] = args.format
        if args.bg_image:
            to_save["default_bg_image"] = args.bg_image
        if args.device:
            to_save["device"] = args.device
        if getattr(args, "internal_res", None) is not None:
            to_save["internal_res"] = args.internal_res
        if getattr(args, "guided_filter", None) is not None:
            to_save["guided_filter"] = args.guided_filter
        if to_save:
            cfg_module.save(to_save)
            console.print(f"💾  Config saved → [underline]{cfg_module.CONFIG_PATH}[/underline]")

    # ── Device + model ───────────────────────────────────────────────────
    device = models.select_device(device_pref)
    console.print(f"🖥  Device: [bold]{device}[/bold]")

    model, model_name = models.load(
        model_type=model_type,
        device=device,
        custom_path=args.model,
    )
    console.print(f"🧠  Model loaded: [bold]{model_name}[/bold]\n")

    # ── Background plate ─────────────────────────────────────────────────
    # RVM doesn't need one, but we still accept one (it will be ignored
    # in the pipeline since RVM takes no bg tensor).
    bg_image = background.resolve(
        cli_bg_image=args.bg_image,
        video_path=str(input_path),
        interactive=True,
    ) if (model_type == "bgmv2" or args.bg_image) else None

    # For RVM with no user-supplied bg, skip the prompt entirely.
    if model_type == "rvm" and args.bg_image is None:
        bg_image = None

    # ── Preview mode ──────────────────────────────────────────────────────
    if args.preview:
        out_folder = pipeline.run_preview(
            input_path=input_path,
            n=args.preview,
            model=model,
            model_type=model_type,
            device=device,
            bg_image=bg_image,
            max_short=max_short,
            internal_res=internal_res,
            guided_filter=guided_filter,
        )
        console.print(f"\n🔍  Preview saved to: [underline]{out_folder}[/underline]")
        sys.exit(0)

    # ── Full pipeline ─────────────────────────────────────────────────────
    pipeline.run(
        input_path=input_path,
        output_path=output_path,
        model=model,
        model_type=model_type,
        model_name=model_name,
        device=device,
        bg_image=bg_image,
        fmt=fmt,
        max_short=max_short,
        internal_res=internal_res,
        guided_filter=guided_filter,
    )

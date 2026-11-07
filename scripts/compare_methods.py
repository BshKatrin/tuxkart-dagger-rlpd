"""Compare explicit completed runs; never scan stopped/extra seeds into results."""

import argparse

from kartstudy.comparison import build_comparison, write_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", required=True, help="Explicit completed run directories"
    )
    parser.add_argument(
        "--output", required=True, help="Fresh ignored analysis directory"
    )
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    try:
        output = write_comparison(
            build_comparison(args.runs), args.output, figures=not args.no_figures
        )
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"Verified five-method comparison: {output}")


if __name__ == "__main__":
    main()

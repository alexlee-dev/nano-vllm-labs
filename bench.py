from __future__ import annotations

import argparse

from nanovllm_labs.bench_specs import run_bench_spec


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lab", type=int, choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--solution", action="store_true")
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> None:
    args, remaining = parse_args(argv)
    if args.lab is None:
        parser = argparse.ArgumentParser(description="Dispatch to a per-lab benchmark entrypoint.")
        parser.add_argument("--lab", type=int, required=True, choices=[1, 2, 3, 4, 5, 6])
        parser.add_argument("--solution", action="store_true")
        parser.print_help()
        return
    run_bench_spec(remaining, lab=args.lab, solution=args.solution)


if __name__ == "__main__":
    main()

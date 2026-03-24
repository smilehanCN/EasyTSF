import argparse

from easytsf.workflow.study import run_study


def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--study", required=True, type=str, help="study config id or YAML path")
    parser.add_argument("-d", "--data_root", default="dataset", type=str, help="data root")
    parser.add_argument("--save_root", default="save", type=str, help="save root")
    parser.add_argument("--accelerator", default="auto", type=str, help="accelerator to use")
    parser.add_argument("--devices", default="auto", type=str, help="device ids/count, e.g. auto, 1, 0,1")
    parser.add_argument("--use_wandb", default=0, type=int, help="use wandb")
    parser.add_argument("--dry_run", default=0, type=int, help="expand study cases without training")
    parser.add_argument("--resume", default=1, type=int, help="skip cases with successful metrics.json")
    parser.add_argument("--fail_fast", default=0, type=int, help="stop after the first failed case")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    runtime_overrides = {
        "data_root": args.data_root,
        "save_root": args.save_root,
        "accelerator": args.accelerator,
        "devices": args.devices,
        "use_wandb": args.use_wandb,
    }
    result = run_study(
        args.study,
        runtime_overrides=runtime_overrides,
        dry_run=bool(args.dry_run),
        resume=bool(args.resume),
        fail_fast=bool(args.fail_fast),
    )
    if result["runs_path"] is not None:
        print("Saved runs report:", result["runs_path"])
        print("Saved summary report:", result["summary_path"])


if __name__ == "__main__":
    main()

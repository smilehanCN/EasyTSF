import argparse
from easytsf.workflow.experiment import (
    add_config_override_args,
    add_shared_runtime_args,
    build_runtime_overrides,
    finalize_runtime_conf,
    load_config,
    parse_config_overrides,
    run_evaluation,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_shared_runtime_args(parser)
    add_config_override_args(parser)
    parser.add_argument("--ckpt_path", default="best", type=str, help="checkpoint path or best/last")
    args = parser.parse_args()

    conf = load_config(args.config, overrides=parse_config_overrides(args.config_overrides))
    conf.update(build_runtime_overrides(args, include_ckpt_path=True))
    run_evaluation(finalize_runtime_conf(conf), ckpt_path=args.ckpt_path)

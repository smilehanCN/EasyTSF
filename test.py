import argparse
from easytsf.experiment import (
    add_shared_runtime_args,
    build_runtime_overrides,
    finalize_runtime_conf,
    load_config,
    run_test,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_shared_runtime_args(parser)
    parser.add_argument("--ckpt_path", default="best", type=str, help="checkpoint path or best/last")
    args = parser.parse_args()

    conf = load_config(args.config)
    conf.update(build_runtime_overrides(args, include_ckpt_path=True))
    run_test(finalize_runtime_conf(conf), ckpt_path=args.ckpt_path)

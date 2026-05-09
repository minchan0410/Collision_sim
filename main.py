import argparse
import os
import yaml
# from pprint import pprint
import numpy as np
import pdb

try:
    from easydict import EasyDict
except ModuleNotFoundError:
    class EasyDict(dict):
        def __getattr__(self, key):
            try:
                return self[key]
            except KeyError as exc:
                raise AttributeError(key) from exc

        def __setattr__(self, key, value):
            self[key] = value



def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of MID')
    parser.add_argument('--config', default='configs/train.yaml')
    parser.add_argument('--dataset', default='')
    return parser.parse_args()


def main():
    # parse arguments and load config
    args = parse_args()
    if not os.path.exists(args.config):
       raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, encoding="utf-8") as f:
       config = yaml.safe_load(f)
    if config is None:
       config = {}

    if args.config:
       config["config"] = args.config
    if args.dataset:
       config["dataset"] = args.dataset
    config["exp_name"] = os.path.splitext(os.path.basename(args.config))[0]
    #pdb.set_trace()
    config = EasyDict(config)

    from mid import MID
    agent = MID(config)

    # keyattr = ["lr", "data_dir", "epochs", "dataset", "batch_size","diffnet"]
    # keys = {}
    # for k,v in config.items():
    #     if k in keyattr:
    #         keys[k] = v
    #
    # pprint(keys)

    sampling = config.get("sampling", "ddpm")
    steps = 100

    if config["eval_mode"]:
        agent.eval(sampling, 100//steps)
    else:
        agent.train()





if __name__ == '__main__':
    main()

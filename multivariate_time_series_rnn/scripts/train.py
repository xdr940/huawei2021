# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from utils.yaml_wrapper import YamlHandler
import argparse
from my_trainer import Trainer



def main(args):
    opts = YamlHandler(args.settings).read_yaml()
    trainer = Trainer(opts,settings=args.settings)#after decoder the yaml file, send the filename to trainer for checkpoints saving
    trainer(opts)
    print('training over')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSfM")
    parser.add_argument("--settings", default='./opts/ctmnt.yaml')
    args = parser.parse_args()
    main(args)

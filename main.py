import argparse

from runner import Runner
from utils import parse_command_line_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='PDSRec', help='Model name')
    parser.add_argument('--sd', type=str, default='C')
    parser.add_argument('--td', type=str, default='O')
    parser.add_argument('--exp_type', type=str, default='srec')

    return parser.parse_known_args()


if __name__ == '__main__':
    args, unparsed_args = parse_args()
    command_line_configs = parse_command_line_args(unparsed_args)
    args_dict = vars(args)

    merged_dict = {**args_dict, **command_line_configs}


    runner = Runner(
        model_name=args.model,
        config_dict= merged_dict
    )
    runner.run()

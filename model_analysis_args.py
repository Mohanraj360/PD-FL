import argparse
import yaml

def arg_set():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default="", help="input yaml config file")
    parser.add_argument('--save_pth', type=str,
                        default="./result_save", help="input yaml config file")
    parser.add_argument('--grid_save_pth', type=str,
                        default="./result_save", help="input yaml config file")
    parser.add_argument('--comment', type=str,
                        default="none", help="leave a comment")
    #     parser.add_argument('--metric', type=str,
    #                         default="new", help="leave a comment")
    parser.add_argument('--thread_rate', type=float,
                        default=0.5, help="use = int(max_thread * t_r) - 1")
    parser.add_argument('--t', type=int,
                        default=15, help="input sample batchsize")
    parser.add_argument('--sleep', type=int,
                        default=0, help="sleep before start")
    parser.add_argument('--check_usage', type=bool, default=True, help="check the usage of cpu before start")
    parser.add_argument('--model', type=str,
                        default="lenet", help="model struct = vgg or lenet")
    parser.add_argument('--number_range', type=list,
                        default=[0,100], help="model number in range like [0,100]")
    # check type list
    parser.add_argument('--round_range', type=list,
                        default=[0,100], help="compute_range like [0,100]")
    parser.add_argument('--pth', type=str,
                        default="/mnt/sda3/docker_space/Code/DBA/saved_models", help="model save path")
    parser.add_argument('--pth_r', type=str,
                        default="save_attack_ub", help="pth save root")
    parser.add_argument('--pth_d', type=str,
                        default="mnist", help="pth dataset")
    parser.add_argument(
        '--pth_m', type=str, default="lenet_iidFalse_num100_C1.0_le1", help="pth modelname")
    parser.add_argument(
        '--pth_s', type=str, default="shard5", help="shared class")
    parser.add_argument('--pth_p', type=str,
                        default="pattern08-27--09-41-17", help="pth pattern")
    parser.add_argument('--skip_metric', type=bool, default=False, help="skip the calculation of vr metric")
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            parser.set_defaults(**yaml.safe_load(f))
            args = parser.parse_args()

    return args
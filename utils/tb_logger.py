import datetime
import json
import os

import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TBLogger:
    def __init__(self, args, exp_label):
        timestamp = datetime.datetime.now().strftime('%H_%M_%S__%d_%m')
        dir_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'logs'))

        try:
            env_name = args.env_name
            seed = args.seed
            algo_name = getattr(args, 'algo_name', exp_label).lower()
        except Exception:
            env_name = args["env_name"]
            seed = args["seed"]
            algo_name = str(args.get("algo_name", exp_label)).lower()

        self.output_name = f'{env_name}__{seed}__{timestamp}'
        self.full_output_folder = os.path.join(dir_path, algo_name, self.output_name)
        os.makedirs(self.full_output_folder, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.full_output_folder)

        print('logging under', self.full_output_folder)

        with open(os.path.join(self.full_output_folder, 'config.json'), 'w') as f:
            try:
                config = {k: v for (k, v) in vars(args).items() if k != 'device'}
            except:
                config = args
            config.update(device=device.type)
            json.dump(config, f, indent=2)

    def add(self, name, value, x_pos):
        self.writer.add_scalar(name, value, x_pos)

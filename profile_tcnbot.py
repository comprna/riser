import torch
from torch.profiler import profile, record_function

from tcn_bot import TCNBot
from utilities import get_config

def main():
    torch.backends.cudnn.enabled = False # TODO: Also try with cuda

    config = get_config('config-tcn-bot.yaml')

    model = TCNBot(config.tcnbot)
    inputs = torch.randn(32, 12048)

    # warm-up (to avoid setup overhead in profiling)
    model(inputs)

    with profile(with_stack=True, profile_memory=True) as prof:
        with record_function('model_inference'):
            model(inputs)
    
    print(prof.key_averages().table(sort_by="cpu_time_total"))


if __name__=="__main__":
    main()
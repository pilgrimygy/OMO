import os
import json
import ast
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def load_config(config_path, alg_name, update_args):
    default_config_path_elements = config_path.split("/")
    default_config_path_elements[-1] = alg_name + ".json"
    default_config_path = os.path.join(*default_config_path_elements)
    with open(default_config_path, 'r') as f:
        default_args_dict = json.load(f)
    with open(config_path,'r') as f:
        args_dict = json.load(f)

    #update args is tpule type, convert to dict type
    update_args_dict = {}
    for update_arg in update_args:
        key, val = update_arg.split("=")
        update_args_dict[key] = ast.literal_eval(val)
    
    #update env specific args to default 
    default_args_dict = update_parameters(default_args_dict, update_args_dict)
    args_dict = merge_dict(default_args_dict, args_dict)
    if 'common' in args_dict:
        for sub_key in args_dict:
            if type(args_dict[sub_key]) == dict:
                args_dict[sub_key] = merge_dict(args_dict[sub_key], default_args_dict['common'], "common")
    return args_dict

def merge_dict(source_dict, update_dict, ignored_dict_name=""):
    for key in update_dict:
        if key == ignored_dict_name:
            continue
        if key not in source_dict:
            #print("\033[32m new arg {}: {}\033[0m".format(key, update_dict[key]))
            source_dict[key] = update_dict[key]
        else:
            assert type(source_dict[key]) == type(update_dict[key])
            if type(update_dict[key]) == dict:
                source_dict[key] = merge_dict(source_dict[key], update_dict[key], ignored_dict_name)
            else:
                print("updated {} from {} to {}".format(key, source_dict[key], update_dict[key]))
                source_dict[key] = update_dict[key]
    return source_dict

def update_parameters(source_args, update_args):
    print("updating args", update_args)
    #command line overwriting case, decompose the path and overwrite the args
    for key_path in update_args:
        target_value = update_args[key_path]
        print("key:{}\tvalue:{}".format(key_path, target_value))
        source_args = overwrite_argument_from_path(source_args, key_path, target_value)
    return source_args


def overwrite_argument_from_path(source_dict, key_path, target_value):
    key_path = key_path.split("/")
    curr_dict = source_dict
    for key in key_path[:-1]:
        if not key in curr_dict:
            #illegal path
            return source_dict
        curr_dict = curr_dict[key]
    final_key = key_path[-1] 
    curr_dict[final_key] = ast.literal_eval(target_value)
    return source_dict


class Logger:
    def __init__(self, log_path, tb_dir="tb_logs", prefix="", warning_level=3, print_to_terminal=True):
        unique_path = self.make_simple_log_path(prefix)
        log_path = os.path.join(log_path, unique_path)
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.tb_writer = SummaryWriter(log_path)
        self.log_file_path = os.path.join(log_path, "logs.txt")
        self.print_to_terminal = print_to_terminal
        self.warning_level = warning_level
        
    def make_simple_log_path(self, prefix):
        now = datetime.now()
        suffix = now.strftime("%m-%d(%H:%M)")
        pid_str = os.getpid()
        return "{}-{}-{}".format(prefix, suffix, pid_str)

    @property
    def log_dir(self):
        return self.log_path
        
    def log_str(self, content, level = 4):
        if level < self.warning_level:
            return
        now = datetime.now()
        time_str = now.strftime("%Y-%m-%d %H:%M:%S")
        if self.print_to_terminal:
            print("\033[32m{}\033[0m:\t{}".format(time_str, content))
        with open(self.log_file_path,'a+') as f:
            f.write("{}:\t{}\n".format(time_str, content))

    def log_var(self, name, val, ite):
        self.tb_writer.add_scalar(name, val, ite)

    def log_str_object(self, name: str, log_dict: dict = None, log_str: str = None):
        if log_dict!=None:
            log_str = json.dumps(log_dict, indent=4)            
        elif log_str!= None:     
            pass
        else:
            assert 0
        if name[-4:] != ".txt":
            name += ".txt"
        target_path = os.path.join(self.log_path, name)
        with open(target_path,'w+') as f:
            f.write(log_str)
        self.log_str("saved {} to {}".format(name, target_path))

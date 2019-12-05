import yaml

def load_config(config_path):
    with open(config_path, 'r') as yml:
        cfg = yaml.safe_load(yml)
    return cfg
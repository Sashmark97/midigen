import os
import sys
import yaml
import shutil
import  argparse
sys.path.append('..')

from midigen.trainer import Trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--yaml_path', help="Path to experiment_config.yaml")
    args = parser.parse_args()

    with open(args.yaml_path) as f:
        trainer = yaml.load(f, yaml.Loader)
    shutil.copy2(args.yaml_path, os.path.join(trainer.save_folder, "config.yml"))

    print('Starting training...')
    trainer.train()
    print('Training completed.')

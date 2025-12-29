import yaml
import shutil
import argparse

from utils import params_from_yaml

parser = argparse.ArgumentParser(
        description="A script that processes a file specified by a path."
)
    
parser.add_argument(
    '--jax',
    action='store_true',
    help='Enable JAX mode'
)

args = parser.parse_args()

params = params_from_yaml("./experiment.yaml")

prefix = ""

if args.jax:
    prefix = "_jax"


print(f"deleting current experiment! {params["name"]}")
experiment = f"./runs{prefix}/run_{params["name"]}"
shutil.rmtree(experiment)
print(f"deleted current experiment {experiment}!")


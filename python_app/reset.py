import yaml
import shutil

from utils import params_from_yaml

params = params_from_yaml("./experiment.yaml")

print(f"deleting current experiment! {params["name"]}")
shutil.rmtree(f"./runs/run_{params["name"]}")
print("deleted current experiment!")


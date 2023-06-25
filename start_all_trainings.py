# %%
from subprocess import Popen
from pathlib import Path

# %%
method_names = ["att", "lstm", "tf", "acs", "2_5d", "3d", "mp", "avg"]
dataset_names = ["organ", "nodule", "fracture", "adrenal", "vessel", "synapse"]

configs = {
    d_name: [f"{d_name}_{m_name}.yaml" for m_name in method_names]
    for d_name in dataset_names
}


# %%
base_path = Path.cwd() / "configs"

for c_list in configs.values():
    for c in c_list:
        c_path = base_path / c
        assert c_path.is_file()
# %%
# %%
commands = {
    k: [
        f"python twoandahalfdimensions/train.py -m -cn {c} general.log_wandb=True general.seed=1,2,3,4".split(
            " "
        )
        # f"python -c \"print('hello_world')\"".split(" ")
        for c in v
    ]
    for k, v in configs.items()
}

# %%
for dataset, command_list in commands.items():
    handles = []
    for cmd in command_list:
        handles.append(Popen(cmd))
    [h.communicate() for h in handles]

# %%

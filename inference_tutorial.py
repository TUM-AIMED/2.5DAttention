# %%
from omegaconf import OmegaConf
from typing import Union
from pathlib import Path
from torch import load


from twoandahalfdimensions.utils.config import Config
from twoandahalfdimensions.utils.setup import setup
from twoandahalfdimensions.utils.data import make_data, make_loader
from twoandahalfdimensions.models.preset_models import make_model_from_config
from twoandahalfdimensions.utils.training_utils import validate


# %%
def make_configs(train_config_path: Union[str, Path]) -> tuple[OmegaConf, Config]:
    base_config = OmegaConf.structured(Config)
    train_config = OmegaConf.load(train_config_path)
    del train_config.defaults
    train_config = OmegaConf.merge(base_config, train_config)
    return train_config


# %%
path_to_model_state_dict = Path("out/pdac_acs_private_e78514dh/final_state.pt")
state_dict = load(path_to_model_state_dict, map_location='cpu')
config = state_dict["config"]
config = OmegaConf.create(config)

# %%
metric_fns, settings, activation_fn = setup(config)
train_ds, val_ds, test_ds = make_data(config)
train_dl, val_dl, test_dl = make_loader(config, (train_ds, val_ds, test_ds))
model = make_model_from_config(config)
model.eval()
model = model.to(**settings)
# %%
model.load_state_dict(state_dict=state_dict["model_weights"])
# %%
test_metrics = validate(
    config,
    test_dl,
    metric_fns,
    model,
    settings,
    activation_fn=activation_fn,
    data_vis_axes=None,
    logging_step=None,
)
# %%
print(test_metrics)

# %%

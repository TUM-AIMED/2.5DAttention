import hydra
from pathlib import Path

from twoandahalfdimensions.models import preset_models


@hydra.main(version_base=None, config_path=Path.cwd() / "configs")
def main(config):
    pass


if __name__ == "__main__":
    main()

import torch
import guidenav
from pathlib import Path

# Change the default torch hub directory to point inside the package
# to enable cahcing of model weights if running in a docker container
module_dir = Path(guidenav.__path__[0])
torch.hub.set_dir(module_dir.parent / "model_weights" / "torch_hub_checkpoints")
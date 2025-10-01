# selavpr.py
import torch
from torchvision import transforms
from .base_model import BaseModel

from selavpr.models import get_model  # You'll need to clone or pip install SeLaVPR
from selavpr.utils.checkpoint import load_checkpoint

class SeLaVPR(BaseModel):
    default_conf = {
        'arch': 'resnet18',
        'backbone_dim': 512,
        'fc_output_dim': 512,
        'checkpoint_path': 'selavpr_resnet18.ckpt',  # change to your actual path
    }
    required_inputs = ['image']

    def _init(self, conf):
        self.model = get_model(
            arch=conf['arch'],
            backbone_dim=conf['backbone_dim'],
            fc_output_dim=conf['fc_output_dim']
        )
        load_checkpoint(self.model, conf['checkpoint_path'])
        self.model.eval()

    def _forward(self, data):
        image = data['image']  # shape: [B, 3, H, W], normalized
        with torch.no_grad():
            desc = self.model(image)
        return {
            'global_descriptor': desc,
        }

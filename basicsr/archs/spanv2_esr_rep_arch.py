"""Training-time re-parameterized SPANV2 for report reproduction.

The public Team 22 model contains already-fused 3x3 convolutions. SPAN's
official training implementation uses a wider 1x1-3x3-1x1 branch plus a 1x1
shortcut and fuses both branches for deployment. This module applies that
training-time parameterization to the otherwise unchanged SPANV2 topology.
"""

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.utils.registry import ARCH_REGISTRY
from .spanv2_esr_arch import OfficialSPANV2ESR


class RepConv3XC(nn.Module):
    """SPAN Conv3XC training branches with exact bias-free deployment fusion."""

    def __init__(self, c_in, c_out, gain1=2, s=1, bias=False):
        super().__init__()
        if bias:
            raise ValueError(
                'SPANV2 REP reproduction only supports bias=False. Bias in the '
                'first 1x1 branch cannot be represented exactly at image borders '
                'after zero-padding fusion.')
        hidden_in = c_in * gain1
        hidden_out = c_out * gain1
        self.stride = s
        self.sk = nn.Conv2d(c_in, c_out, 1, stride=s, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, hidden_in, 1, bias=False),
            nn.Conv2d(hidden_in, hidden_out, 3, stride=s, padding=0, bias=False),
            nn.Conv2d(hidden_out, c_out, 1, bias=False),
        )

    def forward(self, x):
        return self.conv(F.pad(x, (1, 1, 1, 1))) + self.sk(x)

    def get_equivalent_kernel(self):
        """Return the single 3x3 kernel equivalent to both branches."""
        w1 = self.conv[0].weight[:, :, 0, 0]
        w2 = self.conv[1].weight
        w3 = self.conv[2].weight[:, :, 0, 0]
        kernel = torch.einsum('omxy,mi->oixy', w2, w1)
        kernel = torch.einsum('po,oixy->pixy', w3, kernel)
        return kernel + F.pad(self.sk.weight, (1, 1, 1, 1))


def _replace_block_convolutions(block):
    for name in ('c1', 'c2', 'c3'):
        deployed = getattr(block, name).conv
        setattr(
            block,
            name,
            RepConv3XC(
                deployed.in_channels,
                deployed.out_channels,
                gain1=2,
                s=deployed.stride[0],
                bias=deployed.bias is not None,
            ),
        )


@ARCH_REGISTRY.register()
class SPANV2ESRRep(OfficialSPANV2ESR):
    """SPANV2 topology with SPAN-family Conv3XC branches during training."""

    def __init__(self, *args, **kwargs):
        if kwargs.get('bias', False):
            raise ValueError('SPANV2ESRRep requires bias=false')
        super().__init__(*args, **kwargs)
        for index in range(1, 6):
            _replace_block_convolutions(getattr(self, f'block_{index}'))

    def deploy_state_dict(self, keep_vars=False):
        """Build a state dict accepted by the public Team 22 inference model."""
        result = OrderedDict()
        rep_prefixes = {}
        for module_name, module in self.named_modules():
            if isinstance(module, RepConv3XC):
                rep_prefixes[f'{module_name}.'] = module

        for name, value in self.state_dict(keep_vars=keep_vars).items():
            if any(name.startswith(prefix) for prefix in rep_prefixes):
                continue
            result[name] = value
        for prefix, module in rep_prefixes.items():
            kernel = module.get_equivalent_kernel()
            result[f'{prefix}conv.weight'] = kernel if keep_vars else kernel.detach()
        return result

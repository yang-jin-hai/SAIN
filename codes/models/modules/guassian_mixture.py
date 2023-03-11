import torch
import torch.distributions as D
import torch.nn as nn


class GaussianMixture(nn.Module):
    def __init__(self, gmm_components=5):
        super(GaussianMixture, self).__init__()
        self.gmm_para = nn.Parameter(0.0001 * torch.randn(3, gmm_components)) # init
    
    def refresh(self):
        self.pis, self.mus, self.logvars = self.gmm_para

    def build(self):
        self.refresh()
        std = torch.exp(0.5*torch.clamp(self.logvars, -20, 20))
        weights = self.pis.softmax(dim=0)
        mix = D.Categorical(weights)
        comp = D.Normal(self.mus, std)
        self.gmm = D.MixtureSameFamily(mix, comp)
        
    def sample(self, zshape):
        self.build()
        assert (
            self.gmm.component_distribution.has_rsample
        ), "component_distribution attribute should implement rsample() method"

        weights = self.gmm.mixture_distribution._param.repeat(*zshape, 1)
        comp = nn.functional.gumbel_softmax(weights.log(), hard=True)
        samples = self.gmm.component_distribution.rsample(zshape)
        return (comp * samples).sum(dim=-1)

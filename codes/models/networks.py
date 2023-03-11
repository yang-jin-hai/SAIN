import logging
import math

from models.modules.inv_arch import SAINet
from models.modules.subnet_constructor import subnet

logger = logging.getLogger('base')


####################
# define network
####################

def define(opt):
    opt_net = opt['network']
    subnet_type = opt_net['subnet']
    if opt_net['init']:
        init = opt_net['init']
    else:
        init = 'xavier'
    down_num = int(math.log(opt_net['scale'], 2))
    net = SAINet(opt_net['in_nc'], opt_net['out_nc'], subnet(subnet_type, init), opt_net['e_blocks'], opt_net['v_blocks'], down_num, opt_net['gmm_components'])
    return net

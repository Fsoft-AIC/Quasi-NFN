"""This example creates an NFN to process the weight space of a small CNN and output a scalar,
then verifies that the NFN is permutation invariant. This example doesn't train anything; it
is merely a demonstration of how to use the NFN library.
"""

import torch
from torch import nn
from torch.utils.data.dataloader import default_collate
from nfn.common import state_dict_to_tensors, WeightSpaceFeatures, network_spec_from_wsfeat, inner_network_spec_from_ns
from nfn.layers import HNPSLinear, HNPPool, TupleOp, HNPSPool, HNPSMixerLinear, HNPS_SirenLinear

from examples.basic_cnn.helpers import make_cnn, sample_perm_scale, check_perm_scale_symmetry, strip_cnn_struture


# def make_nfn(network_spec_cnn, network_spec_internal, nfn_channels = 4):
#     # return All_layers(network_spec=network_spec, in_channels=in_channels, nfn_channels=nfn_channels)
#     return nn.Sequential(
#         # io_embed: encode the input and output dimensions of the weight space feature
#         HNPS_SirenLinear(network_spec_cnn, network_spec_internal, in_channels=1, 
#                    out_channels = nfn_channels),
#         TupleOp(nn.ReLU()),

#         HNPSMixerLinear(network_spec=network_spec_internal, in_channels=nfn_channels,
#                         out_channels=nfn_channels),
#         TupleOp(nn.ReLU()),
        
#         HNPS_SirenLinear(network_spec_internal, network_spec_cnn, nfn_channels, nfn_channels),
#         TupleOp(nn.ReLU()),
        
#         HNPSPool(network_spec_cnn, nfn_channels=nfn_channels),
#         nn.Flatten(start_dim=-2),
#         nn.Linear(nfn_channels * HNPPool.get_num_outs(network_spec_cnn), 1)
#     )

def make_nfn(network_spec_cnn, network_spec_internal, nfn_channels = 4):
    # return All_layers(network_spec=network_spec, in_channels=in_channels, nfn_channels=nfn_channels)
    return nn.Sequential(
        # io_embed: encode the input and output dimensions of the weight space feature
        HNPS_SirenLinear(network_spec_cnn, network_spec_internal, in_channels=1, 
                   out_channels = nfn_channels),
        TupleOp(torch.sin),

        HNPS_SirenLinear(network_spec_internal, network_spec_cnn, nfn_channels, nfn_channels),
        TupleOp(torch.sin),
        
        HNPSPool(network_spec_cnn, nfn_channels=nfn_channels),
        nn.Flatten(start_dim=-2),
        nn.Linear(nfn_channels * HNPPool.get_num_outs(network_spec_cnn), 1)
    )

@torch.no_grad()
def main():
    print(f"Sanity check: permuting CNN channels preserves CNN behavior: {check_perm_scale_symmetry()}.")

    # Constructed two feature maps, one that is a permutation of the other.
    wts_and_bs, wts_and_bs_perm_scale = [], []
    for _ in range(1000):
        sd = make_cnn().state_dict()
        wts_and_bs.append(state_dict_to_tensors(sd))
        state_dict_tensors_perm_scale = sample_perm_scale(sd)
        wts_and_bs_perm_scale.append(state_dict_to_tensors(state_dict_tensors_perm_scale))

    # Here we manually collate weights and biases (stack into batch dim).
    # When using a dataloader, the collate is done automatically.
    # default_collate output is [2 (weight and bias), num_layer, batch]
    wtfeat = WeightSpaceFeatures(*default_collate(wts_and_bs))
    wtfeat_perm = WeightSpaceFeatures(*default_collate(wts_and_bs_perm_scale))


    in_network_spec = network_spec_from_wsfeat(wtfeat)
    out_network_spec = inner_network_spec_from_ns(in_network_spec)
    nfn = make_nfn(in_network_spec, out_network_spec)
    print(nfn)

    out = nfn(wtfeat)
    out_of_perm = nfn(wtfeat_perm)
    diff = torch.abs(out - out_of_perm)
    #get the number of diff with value greater than 0.1
    diff_count = torch.sum(diff < 0.01)
    print(diff_count)
    print(f"NFN is invariant: {torch.allclose(out, out_of_perm, atol=1e-2)}.")


if __name__ == "__main__":
    main()
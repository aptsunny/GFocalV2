# Copyright (c) Alibaba, Inc. and its affiliates.
# The implementation is also open-sourced by the authors, and available at
# https://github.com/alibaba/lightweight-neural-architecture-search.

from .blocks_basic import (BaseSuperBlock, ConvKXBN, ConvKXBNRELU,
                           network_weight_stupid_init)
from .super_res_k1dwk1 import ResK1DWK1, SuperResK1DWK1
from .super_res_k1kxk1 import ResK1KXK1, SuperResK1KXK1

__all_blocks__ = {
    'ConvKXBN': ConvKXBN,
    'ConvKXBNRELU': ConvKXBNRELU,
    'BaseSuperBlock': BaseSuperBlock,
    'ResK1KXK1': ResK1KXK1,
    'SuperResK1KXK1': SuperResK1KXK1,
    'ResK1DWK1': ResK1DWK1,
    'SuperResK1DWK1': SuperResK1DWK1,
}

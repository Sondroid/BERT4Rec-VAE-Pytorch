from .popular import PopularNegativeSampler
from .random import RandomNegativeSampler


NEGATIVE_SAMPLERS = {
    PopularNegativeSampler.code(): PopularNegativeSampler,
    RandomNegativeSampler.code(): RandomNegativeSampler,
}

def negative_sampler_factory(code, train, eval, item_count, sample_size, mode, save_folder):
    negative_sampler = NEGATIVE_SAMPLERS[code]
    return negative_sampler(train, eval, item_count, sample_size, mode, save_folder)

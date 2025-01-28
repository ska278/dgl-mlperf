import torch as th

def chunkit(seeds, num_parts):
    chunk = seeds.shape[0] // num_parts
    nrem = seeds.shape[0] % num_parts
    rnge = th.tensor([ chunk + 1 if i < nrem else chunk for i in range(num_parts)], dtype=th.int32)
    final_rnge = th.cat((th.tensor([0]), th.cumsum(rnge, 0)))
    return final_rnge, rnge


def accumulate(ten):
    final_rnge = th.cat((th.tensor([0]), th.cumsum(ten, 0)))
    return final_rnge
        
def cumsum(ten):
    return th.cat((th.tensor([0]), th.cumsum(ten, 0)))

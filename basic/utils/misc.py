import numpy as np
import random


try:
    from basic.utils.console.log import ColorPrefeb as CP
except:
    class CPType(type):
        def __getattr__(cls, item):
            return lambda x: x  # 返回恒等函数

    class CP(metaclass=CPType):
        pass


#region ==[random seed]==
# TODO: 训练过程中如果中断了，随机种子如何恢复？
def set_random_seed(seed, rank=0):
    """
    Set random seeds.
    """
    seed = seed + (1 << max(rank, 0)) - 1
    print(f"[{CP.keyword('argparser')}] [rank-{rank}] : Fix random seed <{CP.keyword(seed)}>")

    import os
    os.environ['CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)

    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark = False


def get_random_state():
    """
    Get random state.
    """
    import torch
    return {
        'random': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state()
    }


def set_random_state(seed_state):
    """
    Set random state.
    """
    import torch
    random.setstate(seed_state['random'])
    np.random.set_state(seed_state['numpy'])
    torch.set_rng_state(seed_state['torch'].detach().cpu())
    torch.cuda.set_rng_state(seed_state['cuda'].detach().cpu())
#endregion


def get_random_state_check():
    import torch
    def join_rand(rand_list):
        return ', '.join(f"{r * 100:.0f}" for r in rand_list)

    state = get_random_state()
    rand_rand = [random.random() for _ in range(3)]
    rand_torch = [torch.rand(1).item() for _ in range(3)]
    rand_np = [np.random.rand() for _ in range(3)]
    set_random_state(state)
    return (f"random: [{join_rand(rand_rand)}]; "
            f"torch: [{join_rand(rand_torch)}]; "
            f"numpy: [{join_rand(rand_np)}]")



if __name__ == '__main__':
    set_random_seed(123)
    print(', '.join(f"{random.random():.2f}" for _ in range(10)))

    state = get_random_state()
    print(', '.join(f"{random.random():.2f}" for _ in range(10)))

    set_random_seed(123)
    set_random_state(state)
    print(', '.join(f"{random.random():.2f}" for _ in range(10)))

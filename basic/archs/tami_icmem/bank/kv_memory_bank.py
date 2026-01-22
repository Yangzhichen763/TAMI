import torch
from typing import List


class KeyValueMemoryBank:
    """
    Works for key/value pairs type storage
    e.g., working and long-term memory
    """

    def __init__(self, count_usage: bool):
        self.count_usage = count_usage

        # keys are stored in a single tensor and are shared between groups/objects
        # values are stored as a list indexed by object groups
        self.k = None
        self.v = []

        # shrinkage and selection are also single tensors
        self.s = None
        self.e = None

        # usage
        if self.count_usage:
            self.use_count = None
            self.life_count = None

    def add(self, key, value, shrinkage, selection):
        """
        Args:
            key:        (B, CK, N)
            value:      (B, CV, N)
            shrinkage:  (B, 1, N)
            selection:  (B, CK, N)
        """
        b, _, n = key.shape
        new_count = torch.zeros((b, 1, n), device=key.device, dtype=torch.float32)
        new_life = torch.zeros((b, 1, n), device=key.device, dtype=torch.float32) + 1e-7

        if not isinstance(value, (list, tuple)):
            value = [value]

        # [add] the key
        if self.k is None:
            self.k = key
            self.v = [v for v in value]
            self.s = shrinkage
            self.e = selection
            if self.count_usage:
                self.use_count = new_count
                self.life_count = new_life

        else:
            self.k = torch.cat([self.k, key], -1)
            for i, v in enumerate(value):
                self.v[i] = torch.cat([self.v[i], v], dim=-1)
            if shrinkage is not None:
                self.s = torch.cat([self.s, shrinkage], -1)
            if selection is not None:
                self.e = torch.cat([self.e, selection], -1)
            if self.count_usage:
                self.use_count = torch.cat([self.use_count, new_count], -1)
                self.life_count = torch.cat([self.life_count, new_life], -1)

    def update_usage(self, usage):
        # increase all life count by 1
        # increase use of indexed elements
        if not self.count_usage:
            return

        self.use_count += usage.view_as(self.use_count)
        self.life_count += 1

    def sieve_by_range(self, start: int, end: int, min_size: int):
        # keep only the elements *outside* of this range (with some boundary conditions)
        # i.e., concat (a[:start], a[end:])
        # min_size is only used for values, we do not sieve values under this size
        # (because they are not consolidated)

        if end == 0:
            # negative 0 would not work as the end index!
            self.k = self.k[..., :start]
            if self.count_usage:
                self.use_count = self.use_count[..., :start]
                self.life_count = self.life_count[..., :start]
            if self.s is not None:
                self.s = self.s[..., :start]
            if self.e is not None:
                self.e = self.e[..., :start]

            for i in range(self.num_values):
                if self.v[i].shape[-1] >= min_size:
                    self.v[i] = self.v[i][..., :start]
        else:
            self.k = torch.cat([self.k[..., :start], self.k[..., end:]], -1)
            if self.count_usage:
                self.use_count = torch.cat([self.use_count[..., :start], self.use_count[..., end:]], -1)
                self.life_count = torch.cat([self.life_count[..., :start], self.life_count[..., end:]], -1)
            if self.s is not None:
                self.s = torch.cat([self.s[..., :start], self.s[..., end:]], -1)
            if self.e is not None:
                self.e = torch.cat([self.e[..., :start], self.e[..., end:]], -1)

            for i in range(self.num_values):
                if self.v[i].shape[-1] >= min_size:
                    self.v[i] = torch.cat([self.v[i][..., :start], self.v[i][..., end:]], -1)

    def remove_obsolete_features(self, max_size: int):
        # normalize with life duration
        usage = self.get_usage().flatten()

        values, _ = torch.topk(usage, k=(self.size - max_size), largest=False, sorted=True)
        survived = (usage > values[-1])

        self.k = self.k[..., survived]
        self.s = self.s[..., survived] if self.s is not None else None
        # Long-term memory does not store ek so this should not be needed
        self.e = self.e[..., survived] if self.e is not None else None
        if self.num_values > 1:
            raise NotImplementedError("""The current data structure does not support feature removal with 
            multiple object groups (e.g., some objects start to appear later in the video)
            The indices for "survived" is based on keys but not all values are present for every key
            Basically we need to remap the indices for keys to values
            """)
        for i in range(self.num_values):
            self.v[i] = self.v[i][..., survived]

        self.use_count = self.use_count[..., survived]
        self.life_count = self.life_count[..., survived]

    def get_usage(self):
        # return normalized usage
        if not self.count_usage:
            raise RuntimeError('I did not count usage!')
        else:
            usage = self.use_count / self.life_count
            return usage

    def get_all_sliced(self, start: int, end: int):
        # return k, sk, ek, usage in order, sliced by start and end

        if end == 0:
            # negative 0 would not work as the end index!
            k = self.k[:, :, start:]
            sk = self.s[:, :, start:] if self.s is not None else None
            ek = self.e[:, :, start:] if self.e is not None else None
            usage = self.get_usage()[:, :, start:]
        else:
            k = self.k[:, :, start:end]
            sk = self.s[:, :, start:end] if self.s is not None else None
            ek = self.e[:, :, start:end] if self.e is not None else None
            usage = self.get_usage()[:, :, start:end]

        return k, sk, ek, usage

    def get_v_size(self, ni: int):
        return self.v[ni].shape[-1]

    def engaged(self):
        return self.k is not None

    @property
    def size(self):
        if self.k is None:
            return 0
        else:
            return self.k.shape[-1]

    @property
    def num_values(self):
        return len(self.v)

    @property
    def key(self):
        return self.k

    @property
    def value(self):
        return self.v

    @property
    def shrinkage(self):
        return self.s

    @property
    def selection(self):
        return self.e


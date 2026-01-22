import torch
import warnings

import torchvision.transforms

from .kv_memory_bank import KeyValueMemoryBank
from .memory_util import *


class MemoryManager:
    """
    Manages all three memory stores and the transition between working/long-term memory
    """

    def __init__(
            self,
            hidden_dim, top_k,
            min_mid_term_frames=1, max_mid_term_frames=3,
            enable_long_term=True, max_long_term_elements=1024, num_long_term_prototypes=32,
            enable_long_term_count_usage=True,
    ):
        self.hidden_dim = hidden_dim
        self.top_k = top_k

        self.enable_long_term = enable_long_term
        self.enable_long_term_usage = enable_long_term_count_usage
        if self.enable_long_term:
            assert max_mid_term_frames >= min_mid_term_frames >= 1, \
                f"Invalid mid-term memory frame settings, got min {min_mid_term_frames}, max {max_mid_term_frames}"
            self.min_mt_frames = min_mid_term_frames
            self.max_mt_frames = max_mid_term_frames
            self.num_long_term_prototypes = num_long_term_prototypes
            self.max_long_elements = max_long_term_elements

        # dimensions will be inferred from input later
        self.CK = self.CV = None
        self.H = self.W = None

        # The hidden state will be stored in a single tensor for all objects
        # B x num_objects x CH x H x W
        self.hidden = None

        self.work_mem = KeyValueMemoryBank(count_usage=self.enable_long_term)
        if self.enable_long_term:
            self.long_mem = KeyValueMemoryBank(count_usage=self.enable_long_term_usage)

        self.reset_config = True

    def update_config(
            self,
            hidden_dim, top_k,
            min_mid_term_frames, max_mid_term_frames,
            enable_long_term, max_long_term_elements, num_long_term_prototypes,
            enable_long_term_count_usage,
    ):
        self.reset_config = True

        self.hidden_dim = hidden_dim
        self.top_k = top_k

        assert self.enable_long_term == enable_long_term, 'cannot update enable_long_term'
        assert self.enable_long_term_usage == enable_long_term_count_usage, 'cannot update enable_long_term_usage'

        if self.enable_long_term:
            assert max_mid_term_frames >= min_mid_term_frames >= 1, \
                f"Invalid mid-term memory frame settings, got min {min_mid_term_frames}, max {max_mid_term_frames}"
            self.min_mt_frames = min_mid_term_frames
            self.max_mt_frames = max_mid_term_frames
            self.num_long_term_prototypes = num_long_term_prototypes
            self.max_long_elements = max_long_term_elements

    def _readout(self, affinity, v):
        # this function is for a single object group
        return v @ affinity

    def match_memory(self, query_key, selection=None, softmax_func=None):
        # query_key:  (B, C_k, H, W)
        # selection:  (B, C_k, H, W)
        # out:        (B, C_v, H, W)
        b, _, h, w = query_key.shape

        if softmax_func is None:
            softmax_func = do_softmax

        query_key = query_key.flatten(start_dim=2)
        selection = selection.flatten(start_dim=2) if selection is not None else None

        """
        Memory readout using keys
        """
        if self.enable_long_term and self.long_mem.engaged():
            # Use long-term memory
            long_mem_size = self.long_mem.size
            memory_key = torch.cat([self.long_mem.key, self.work_mem.key], -1)
            shrinkage = torch.cat([self.long_mem.shrinkage, self.work_mem.shrinkage], -1) if selection is not None else None

            similarity = get_similarity(memory_key, query_key, shrinkage, selection)
            work_mem_similarity = similarity[:, long_mem_size:]
            long_mem_similarity = similarity[:, :long_mem_size] # 前半部分是 long-term 的

            # get the usage with the first group
            # the first group always have all the keys valid
            affinity, usage = softmax_func(
                torch.cat([long_mem_similarity, work_mem_similarity], 1),
                top_k=self.top_k, inplace=True, return_usage=True
            )
            # print(f"affinity shape: {affinity.shape}, usage shape: {usage.shape}")

            all_memory_value = [
                torch.cat([long_mem_value, work_mem_value], -1)
                for long_mem_value, work_mem_value in zip(self.long_mem.value, self.work_mem.value)
            ]

            """
            Record memory usage for working and long-term memory
            """
            # ignore the index return for long-term memory
            work_usage = usage[:, long_mem_size:]
            self.work_mem.update_usage(work_usage.flatten())

            if self.enable_long_term_usage:
                # ignore the index return for working memory
                long_usage = usage[:, :long_mem_size]
                self.long_mem.update_usage(long_usage.flatten())
        else:
            # No long-term memory
            similarity = get_similarity(self.work_mem.key, query_key, self.work_mem.shrinkage, selection)

            if self.enable_long_term:
                affinity, usage = softmax_func(similarity, top_k=self.top_k, return_usage=True)
                # Record memory usage for working memory
                self.work_mem.update_usage(usage.flatten())
            else:
                affinity = softmax_func(similarity, top_k=self.top_k, return_usage=False)
            all_memory_value = self.work_mem.value

        # Readout memory
        readout_mem = [
            self._readout(affinity, memory_value).view(b, self.CV, h, w)
            for memory_value in all_memory_value
        ]
        return readout_mem

    def add_memory(self, key, value, shrinkage=None, selection=None):
        # key:      (1, CK, H, W)
        # value:    (1, CV, H, W)
        # objects contain a list of object indices
        if self.H is None or self.reset_config:
            self.reset_config = False
            self.H, self.W = key.shape[-2:]
            self.HW = self.H * self.W
            if self.enable_long_term:
                # convert from num. frames to num. nodes
                self.min_work_elements = max(self.HW, self.min_mt_frames * self.HW)
                self.max_work_elements = self.max_mt_frames * self.HW

        # key:   (1, C, N)
        # value: (1, C, N)
        key = key.flatten(start_dim=2)
        shrinkage = shrinkage.flatten(start_dim=2) if shrinkage is not None else None
        value = value.flatten(start_dim=2)

        self.CK = key.shape[1]
        self.CV = value.shape[1]

        if selection is not None:
            if not self.enable_long_term:
                warnings.warn('the selection factor is only needed in long-term mode', UserWarning)
            selection = selection.flatten(start_dim=2)

        self.work_mem.add(key, value, shrinkage, selection)

        # long-term memory cleanup
        if self.enable_long_term:
            # Do memory compressed if needed
            if self.work_mem.size >= self.max_work_elements:
                # Remove obsolete features if needed
                if self.long_mem.size > (self.max_long_elements - self.num_long_term_prototypes):
                    self.long_mem.remove_obsolete_features(self.max_long_elements - self.num_long_term_prototypes)

                self.compress_features()

    def add_memory_without_compress(self, key, value, shrinkage=None, selection=None):
        # key:      (1, CK, H, W)
        # value:    (1, CV, H, W)
        # objects contain a list of object indices
        if self.H is None or self.reset_config:
            self.reset_config = False
            self.H, self.W = key.shape[-2:]
            self.HW = self.H * self.W
            if self.enable_long_term:
                # convert from num. frames to num. nodes
                self.min_work_elements = max(self.HW, self.min_mt_frames * self.HW)
                self.max_work_elements = self.max_mt_frames * self.HW

        # key:   (1, C, N)
        # value: (1, C, N)
        key = key.flatten(start_dim=2)
        shrinkage = shrinkage.flatten(start_dim=2) if shrinkage is not None else None
        value = value.flatten(start_dim=2)

        self.CK = key.shape[1]
        self.CV = value.shape[1]

        if selection is not None:
            if not self.enable_long_term:
                warnings.warn('the selection factor is only needed in long-term mode', UserWarning)
            selection = selection.flatten(start_dim=2)

        self.work_mem.add(key, value, shrinkage, selection)

    def create_hidden_state(self, batch_size, size_as):
        h, w = size_as.shape[-2:]
        if self.hidden is None:
            self.hidden = torch.zeros((batch_size, self.hidden_dim, h, w), device=size_as.device)

    def set_hidden(self, hidden):
        self.hidden = hidden

    def get_hidden(self):
        return self.hidden

    def compress_features(self):
        HW = self.HW
        candidate_value = []
        total_work_mem_size = self.work_mem.size
        for value in self.work_mem.value:
            # Some object groups might be added later in the video
            # So not all keys have values associated with all objects
            # We need to keep track of the key->value validity
            mem_size_in_this_group = value.shape[-1]
            if mem_size_in_this_group >= total_work_mem_size:
                # full LT
                candidate_value.append(
                    value[:, :, HW:]
                    if -self.min_work_elements + HW >= 0
                    else value[:, :, HW:-self.min_work_elements + HW]
                )
            else:
                # mem_size is smaller than total_work_mem_size, but at least HW
                assert HW <= mem_size_in_this_group < total_work_mem_size
                if mem_size_in_this_group > self.min_work_elements + HW:
                    # part of this object group still goes into LT
                    candidate_value.append(
                        value[:, :, HW:]
                        if -self.min_work_elements + HW == 0
                        else value[:, :, HW:-self.min_work_elements + HW]
                    )
                else:
                    # this object group cannot go to the LT at all
                    candidate_value.append(None)

        if len(candidate_value) > 0:
            # perform memory consolidation
            prototype_key, prototype_value, prototype_shrinkage = self.consolidation(
                *self.work_mem.get_all_sliced(HW, -self.min_work_elements + HW),
                candidate_value
            )

            # remove consolidated working memory
            self.work_mem.sieve_by_range(HW, -self.min_work_elements + HW, min_size=self.min_work_elements + HW)

            # add to long-term memory
            self.long_mem.add(prototype_key, prototype_value, prototype_shrinkage, selection=None)

    def consolidation(self, candidate_key, candidate_shrinkage, candidate_selection, usage, candidate_value):
        # keys:     (1, C, N)
        # values:   (1, C, N)
        N = candidate_key.shape[-1]

        # find the indices with max usage
        k = min(self.num_long_term_prototypes, N)
        _, max_usage_indices = torch.topk(usage, k=k, dim=-1, sorted=True)
        prototype_indices = max_usage_indices.flatten()

        # Prototypes are invalid for out-of-bound groups
        validity = [prototype_indices >= (N - v.shape[-1]) if v is not None else None for v in candidate_value]

        prototype_key = candidate_key[:, :, prototype_indices]
        prototype_selection = candidate_selection[:, :, prototype_indices] if candidate_selection is not None else None

        """
        Potentiation step
        """
        similarity = get_similarity(candidate_key, prototype_key, candidate_shrinkage, prototype_selection)

        # convert similarity to affinity
        # need to do it group by group since the softmax normalization would be different
        affinity = [
            do_softmax(similarity[:, -v.shape[-1]:, validity[i]]) if v is not None else None
            for i, v in enumerate(candidate_value)
        ]

        # some values can be have all False validity. Weed them out.
        affinity = [
            aff if aff is None or aff.shape[-1] > 0 else None for aff in affinity
        ]

        # readout the values
        prototype_value = [
            self._readout(affinity[i], v) if affinity[i] is not None else None
            for i, v in enumerate(candidate_value)
        ]

        # readout the shrinkage term
        prototype_shrinkage = self._readout(affinity[0], candidate_shrinkage) if candidate_shrinkage is not None else None

        return prototype_key, prototype_value, prototype_shrinkage

    def get_ref(self, batch_size, num_ref_frames=None, indices=None, random=True, random_mode='frame'):
        if num_ref_frames is None:
            num_ref_frames = self.top_k

        t = self.work_mem.size
        if t <= num_ref_frames * self.HW:
            ref_keys = self.work_mem.key[..., :t]
            ref_values = self.work_mem.value[0][..., :t]
            ref_shrinkage = self.work_mem.shrinkage[..., :t] if self.work_mem.shrinkage is not None else None
            indices = torch.arange(t, device=self.work_mem.key.device).unsqueeze(0).expand(batch_size, -1)
        else:
            if not random and indices is None:
                ref_keys = self.work_mem.key[..., -num_ref_frames * self.HW:]
                ref_values = self.work_mem.value[0][..., -num_ref_frames * self.HW:]
                ref_shrinkage = self.work_mem.shrinkage[..., -num_ref_frames * self.HW:] if self.work_mem.shrinkage is not None else None
            else:
                if indices is None:
                    random_mode: str = random_mode
                    if random_mode == 'pixel':
                        perm = torch.randperm(t - 1, device=self.work_mem.key.device)[:batch_size * num_ref_frames * self.HW - 1] + 1
                        perm = perm.view(batch_size, -1)
                        filler_one = torch.zeros(batch_size, 1, dtype=torch.int64, device=self.work_mem.key.device)
                        indices = torch.cat([filler_one, perm], dim=1)
                    elif random_mode.startswith('frame'):
                        HW = self.HW
                        t = self.work_mem.size
                        assert t % HW == 0, f"memory size t={t} must be divisible by HW={HW} for frame-level sampling"
                        num_frames = t // HW

                        if 'include_first' in random_mode:
                            k = max(num_ref_frames - 1, 0)
                            if num_frames <= 1 or k == 0:
                                frame_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=self.work_mem.key.device)
                            else:
                                perm = torch.randperm(num_frames - 1, device=self.work_mem.key.device)[: batch_size * k] + 1
                                perm = perm.view(batch_size, k)
                                first = torch.zeros(batch_size, 1, dtype=torch.long, device=self.work_mem.key.device)
                                frame_ids = torch.cat([first, perm], dim=1)
                        else:
                            k = num_ref_frames
                            perm = torch.randperm(num_frames, device=self.work_mem.key.device)[: batch_size * k]
                            frame_ids = perm.view(batch_size, k)

                        base = (frame_ids * HW).unsqueeze(-1)
                        offsets = torch.arange(HW, device=self.work_mem.key.device).view(1, 1, HW)
                        indices = (base + offsets).reshape(batch_size, -1)

                def gather_x(x):
                    return torch.gather(x, 2, indices.unsqueeze(1).expand(-1, x.size(1), -1))

                ref_keys = gather_x(self.work_mem.key)
                ref_values = gather_x(self.work_mem.value[0])
                ref_shrinkage = gather_x(self.work_mem.shrinkage) if self.work_mem.shrinkage is not None else None

        return ref_keys, ref_values, ref_shrinkage, indices

    def reset(self):
        self.hidden = None
        self.work_mem = KeyValueMemoryBank(count_usage=self.enable_long_term)
        if self.enable_long_term:
            self.long_mem = KeyValueMemoryBank(count_usage=self.enable_long_term_usage)

    def size(self, with_long_term=True):
        if self.enable_long_term and with_long_term:
            return self.work_mem.size + self.long_mem.size
        else:
            return self.work_mem.size

    def is_empty(self):
        return self.size() == 0
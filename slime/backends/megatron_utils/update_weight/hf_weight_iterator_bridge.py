import dataclasses
import re

import torch

from slime.utils import megatron_bridge_utils
from slime.utils.iter_utils import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge
        import slime_plugins.megatron_bridge  # noqa: F401

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint)

    def get_hf_weight_chunks(self, megatron_local_weights):
        # TODO support quantization (e.g. modify megatron-bridge to provide megatron param name)
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        
        # Handle EP: gather expert weights across EP ranks before conversion
        # megatron-bridge adjusts expert indices in names but doesn't gather weights across EP
        renamed_megatron_local_weights = self._gather_expert_weights_across_ep(renamed_megatron_local_weights)
        
        with megatron_bridge_utils.patch_megatron_model(self.model):
            conversion_tasks = self._bridge.get_conversion_tasks(self.model)
            conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)

            named_weights = self._bridge.export_hf_weights(self.model, cpu=False, conversion_tasks=conversion_tasks)

            named_weights = (
                (
                    hf_param_name,
                    postprocess_hf_param(
                        args=self.args,
                        megatron_param_name=megatron_param_name,
                        hf_param_name=hf_param_name,
                        param=weight,
                    ),
                )
                for hf_param_name, weight, megatron_param_name in named_weights
            )

            yield from chunk_named_params_by_size(named_weights, chunk_size=self.args.update_weight_buffer_size)

    def _gather_expert_weights_across_ep(self, megatron_local_weights):
        """
        Gather expert weights from all EP ranks so each rank has all experts.
        Uses memory-efficient approach: gather to CPU first, then move to CUDA only when needed.
        megatron-bridge's _megatron_local_name_to_global only adjusts indices in names,
        but doesn't gather the actual weights across EP ranks.
        """
        try:
            import torch.distributed as dist
            from megatron.core import mpu
        except ImportError:
            return megatron_local_weights
        
        ep_size = mpu.get_expert_model_parallel_world_size()
        if ep_size <= 1:
            return megatron_local_weights
        
        ep_group = mpu.get_expert_model_parallel_group()
        ep_group_ranks = dist.get_process_group_ranks(ep_group)
        
        # Separate expert and non-expert weights
        expert_weights = {}
        non_expert_weights = {}
        
        for name, param in megatron_local_weights.items():
            if ".experts." in name:
                expert_weights[name] = param
            else:
                non_expert_weights[name] = param
        
        # Gather expert weight metadata (names and shapes) from all EP ranks
        local_expert_metadata = {
            name: (param.shape, param.dtype, param.device) for name, param in expert_weights.items()
        }
        all_expert_metadata = [None] * ep_size
        dist.all_gather_object(all_expert_metadata, local_expert_metadata, group=ep_group)
        
        # Group expert weights by layer to process in batches and reduce peak memory
        # Extract layer number from name for grouping
        def get_layer_key(name):
            match = re.search(r'\.layers\.(\d+)\.', name)
            return int(match.group(1)) if match else -1
        
        # Group metadata by layer
        layer_groups = {}
        for ep_rank_idx, rank_metadata in enumerate(all_expert_metadata):
            for name, (shape, dtype, device) in rank_metadata.items():
                layer_key = get_layer_key(name)
                if layer_key not in layer_groups:
                    layer_groups[layer_key] = []
                layer_groups[layer_key].append((ep_rank_idx, name, shape, dtype))
        
        # Process layer by layer to reduce peak memory
        # This reduces peak memory from (all experts) to (one layer's experts)
        gathered_expert_weights = {}
        
        for layer_key in sorted(layer_groups.keys()):
            layer_experts = layer_groups[layer_key]
            
            # Gather experts for this layer
            for ep_rank_idx, name, shape, dtype in layer_experts:
                src_rank = ep_group_ranks[ep_rank_idx]
                expert_offset = ep_rank_idx * self.args.num_experts // ep_size
                
                # Adjust expert index to global
                adjusted_name = self._adjust_expert_index_to_global(name, expert_offset)
                
                # Get or create tensor on CUDA (required for NCCL)
                if dist.get_rank() == src_rank:
                    # This rank owns the weight - ensure on CUDA
                    weight = expert_weights[name]
                    if not weight.is_cuda:
                        weight = weight.cuda()
                else:
                    # Create empty tensor on CUDA to receive the weight
                    weight = torch.empty(shape, dtype=dtype, device=torch.cuda.current_device())
                
                # Broadcast weight from source rank
                dist.broadcast(weight, src=src_rank, group=ep_group)
                gathered_expert_weights[adjusted_name] = weight
            
            # Clear intermediate variables and free GPU memory after each layer
            # This helps reduce peak memory usage
            del layer_experts
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Combine non-expert and gathered expert weights
        result = non_expert_weights.copy()
        result.update(gathered_expert_weights)
        return result
    
    def _adjust_expert_index_to_global(self, name, expert_offset):
        """
        Adjust expert index from local to global based on expert_offset.
        Pattern: vp_stages.{vp}.decoder.layers.{layer}.mlp.experts.{component}.weight{expert_idx}
        """
        expert_pattern = r"(.*\.mlp\.experts\..*\.weight)(\d+)$"
        match = re.match(expert_pattern, name)
        
        if match:
            prefix, local_idx = match.groups()
            global_idx = int(local_idx) + expert_offset
            return f"{prefix}{global_idx}"
        
        return name


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    def _handle_one(task):
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        assert (
            weight_dict_key in new_weight_dict
        ), f"{weight_dict_key=} not in new_weight_dict ({task.vp_stage=}, {task.param_name=}, {list(new_weight_dict)=})"

        new_param_weight = new_weight_dict[weight_dict_key]
        # Move to CUDA if not already there (handles CPU offloaded weights)
        if not new_param_weight.is_cuda:
            new_param_weight = new_param_weight.cuda()
        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)

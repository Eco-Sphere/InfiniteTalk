import logging
import torch
from torch import Tensor
try:
    import torch_npu
    npu_available = True
except:
    npu_available = False

import torch.distributed as dist
import math
import os
from yunchang import LongContextAttention
try:
    from yunchang.kernels import AttnType
except ImportError:
    raise ImportError("Please install yunchang 0.6.0 or later")
from typing import Any
if npu_available:
    # from ..distributed.parallel_mgr import get_sp_group
    from ..distributed.comm import all_to_all_4D
    from wan.utils.rainfusion import Rainfusion
    from wan.utils.rainfusion_blockwise import Rainfusion_blockwise
    # Rainfusion = None
    from mindiesd import attention_forward
    from mindiesd.layers.flash_attn.ascend_laser_attention import AscendLaserAttention
    from mindiesd.layers.flash_attn.ascend_laser_preprocess import la_preprocess, AscendLaserPreprocess

    from mindiesd.layers import _custom_ops as ops
else:
    from xfuser.core.distributed import get_sp_group
    from xfuser.core.comm import all_to_all_4D
    Rainfusion = None
    attention_forward = None

logger = logging.getLogger(__name__)
MAX_TOKEN = 2147483647

# Monkey patch for AscendLaserAttention.forward_attn_bnsd
if npu_available:
    original_forward_attn_bnsd = AscendLaserAttention.forward_attn_bnsd
    original_forward_preprocess = AscendLaserPreprocess.forward_preprocess

    @classmethod
    def patched_forward_attn_bnsd(cls, attn_param, query, key, value, mask=None, scale=None):
        #logging.info(f"patched_forward_attn_bnsd enter")
        head_first = attn_param.head_first
        new_query, new_key, new_value = la_preprocess(query, key, value, align_len=256)
        pre_tokens = MAX_TOKEN
        if attn_param.kv_seqlen % 256 != 0:
            pre_tokens = (attn_param.kv_seqlen // 256 + 1) * 256 - attn_param.kv_seqlen

        _, output1 = ops.laser_attention(
            new_query, new_key, new_value, None, None, None,
            scale, attn_param.head_num, "BNSD", 1.0, pre_tokens, 1, True
        )
        out = AscendLaserAttention.la_postprocess_output(output1, query.dtype, attn_param.q_seqlen, attn_param.head_dim)

        if not head_first:
            out = out.transpose(1, 2)
        return out
    
    AscendLaserAttention.forward_attn_bnsd = patched_forward_attn_bnsd
    logger.info("Monkey patched AscendLaserAttention.forward_attn_bnsd with la_preprocess")

    @classmethod
    def patched_forward_preprocess(cls,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            align_len: int = 256
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ParametersInvalid("LA_preprocess input must 4D tensor")
        batch_size, seq_len, head_num, head_dim = query.shape
        original_dtype = query.dtype

        out_query, out_key, out_value = ops.laser_attention_preprocess(
            query, key, value, align_len
        )
        return out_query, out_key, out_value

    AscendLaserPreprocess.forward_preprocess = patched_forward_preprocess
    logger.info("Monkey patched AscendLaserPreprocess.forward_preprocess with la_preprocess")

from xfuser.core.distributed import (
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    get_sp_group,
)

class xFuserLongContextAttention(LongContextAttention):
    ring_impl_type_supported_kv_cache = ["basic"]

    def __init__(
        self,
        args: Any = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "basic",
        use_pack_qkv: bool = False,
        use_kv_cache: bool = False,
        attn_type: AttnType = AttnType.FA,
        rainfusion_config=None,
    ) -> None:
        """
        Arguments:
            scatter_idx: int = 2, the scatter dimension index for Ulysses All2All
            gather_idx: int = 1, the gather dimension index for Ulysses All2All
            ring_impl_type: str = "basic", the ring implementation type, currently only support "basic"
            use_pack_qkv: bool = False, whether to use pack qkv in the input
            use_kv_cache: bool = False, whether to use kv cache in the attention layer, which is applied in PipeFusion.
        """
        super().__init__(
            scatter_idx=scatter_idx,
            gather_idx=gather_idx,
            ring_impl_type=ring_impl_type,
            use_pack_qkv=use_pack_qkv,
            attn_type = attn_type,
        )
        self.use_kv_cache = use_kv_cache
        if (
            use_kv_cache
            and ring_impl_type not in self.ring_impl_type_supported_kv_cache
        ):
            raise RuntimeError(
                f"ring_impl_type: {ring_impl_type} do not support SP kv cache."
            )
        self.world_size = dist.get_world_size()
        self.args = args
        self.video_size = ['480*832', '832*480', '480*720', '720*480', '1024*1024']

        self.algo = int(os.getenv('ALGO', 0))
        self.algo = 1
        # TODO: Args
        """
        if self.args.size in self.video_size:
            self.use_all_head = True
        else:
            self.use_all_head = False
        """
        self.use_all_head = True
        
        self.ulysses_pg = get_sp_group().ulysses_group
        self.ring_pg = get_sp_group().ring_group

        if Rainfusion:
            # print("Rainfusion True")
            self.rainfusion_config = rainfusion_config
            self.rainfusion_fa = None
            if self.rainfusion_config is not None:
                if rainfusion_config["type"] == "v1":
                    self.rainfusion_fa = Rainfusion(
                        grid_size=rainfusion_config["grid_size"],
                        skip_timesteps=rainfusion_config["skip_timesteps"],
                        sparsity=rainfusion_config["sparsity"],
                    )
                else:
                    #logging.info(f"====== xFuserLongContextAttention enter rainfusion v2")
                    self.rainfusion_fa_blockwise = Rainfusion_blockwise(
                        grid_size=rainfusion_config["grid_size"],
                        pool_size=128,
                        sparsity=rainfusion_config["sparsity"],
                        skip_timesteps=rainfusion_config["skip_timesteps"],
                        txt_len=0,
                    )
            # print(f"self.rainfusion_fa: {self.rainfusion_fa}")
        else:
            self.rainfusion_config = None
            self.rainfusion_fa = None

    def forward(
        self,
        attn,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        joint_tensor_query=None,
        joint_tensor_key=None,
        joint_tensor_value=None,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        joint_strategy="none",
        scale=None,
        t_idx=0,
        b_idx=0,
    ) -> Tensor:
        """forward

        Arguments:
            attn (Attention): the attention module
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args,
            joint_tensor_query: Tensor = None, a replicated tensor among processes appended to the front or rear of query, depends the joint_strategy  
            joint_tensor_key: Tensor = None, a replicated tensor among processes appended to the front or rear of key, depends the joint_strategy
            joint_tensor_value: Tensor = None, a replicated tensor among processes appended to the front or rear of value, depends the joint_strategy,
            *args: the args same as flash_attn_interface
            joint_strategy: str = "none", the joint strategy for joint attention, currently only support "front" and "rear"

        Returns:
            * output (Tensor): context output
        """

        query_layer = all_to_all_4D(input_=query, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
        key_layer = all_to_all_4D(input_=key, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)
        value_layer = all_to_all_4D(input_=value, scatter_idx=2, gather_idx=1, group=self.ulysses_pg)

        # print(f"query_layer.shape: {query_layer.shape}")
        # print(f"key_layer.shape: {key_layer.shape}")
        # print(f"value_layer.shape: {value_layer.shape}")

        if get_sp_group().ring_world_size > 1:
            ring_size = get_sp_group().ring_world_size
            b, s, n, d = key_layer.shape
            k_full = torch.empty([ring_size, b, s, n, d], dtype=query_layer.dtype, device=query_layer.device)
            dist.all_gather_into_tensor(k_full, key_layer, group=self.ring_pg)
            key_layer = k_full.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d)

            v_full = torch.empty([ring_size, b, s, n, d], dtype=query_layer.dtype, device=query_layer.device)
            dist.all_gather_into_tensor(v_full, value_layer, group=self.ring_pg)
            value_layer = v_full.permute(1, 0, 2, 3, 4).reshape(b, -1, n, d)

        if self.rainfusion_config is not None:
            if self.rainfusion_config["type"] == "v1":
                out = self.rainfusion_fa(
                    query_layer,
                    key_layer,
                    value_layer,
                    atten_mask_all=self.rainfusion_config["atten_mask_all"],
                    text_len=0,
                    t_idx=t_idx,
                )
            else:
                out, _ = self.rainfusion_fa_blockwise(
                    query_layer,
                    key_layer,
                    value_layer,
                    t_b_idx=[t_idx, b_idx],
                    base_blockmask=None,
                )
        elif self.use_all_head:
            if attention_forward:
                if self.algo == 0:
                    # print("attention_forward  fused_attn_score")
                    out = attention_forward(query_layer, key_layer, value_layer,
                                            opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
                elif self.algo == 1:
                    # print("attention_forward  LA")
                    out = attention_forward(query_layer, key_layer, value_layer,
                                            opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
                else:
                    raise ValueError(f"select flash attention algorithm only support 0, 1, but got {self.algo}")
            else:
                raise ValueError("attention_forward is not available")
        else:
            query_layer_list = query_layer.split(1, dim=2)
            key_layer_list = key_layer.split(1, dim=2)
            value_layer_list = value_layer.split(1, dim=2)
            output = []
            for_loop = query_layer.shape[2]
            for i in range(for_loop):
                if self.algo == 0:
                    out = attention_forward(query_layer_list[i], key_layer_list[i], value_layer_list[i],
                                        opt_mode="manual", op_type="fused_attn_score", layout="BNSD")
                elif self.algo == 1:
                    # print("attention_forward  not use_all_head LA")
                    out = attention_forward(query_layer_list[i], key_layer_list[i], value_layer_list[i],
                                        opt_mode="manual", op_type="ascend_laser_attention", layout="BNSD")
                else:
                    raise ValueError(f"select flash attention algorithm only support 0, 1, but got f{self.algo}")

                output.append(out)
            out = torch.cat(output, dim=2)

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = all_to_all_4D(input_=context_layer, scatter_idx=1, gather_idx=2, group=self.ulysses_pg)

        return output


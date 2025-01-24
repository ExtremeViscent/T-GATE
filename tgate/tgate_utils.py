import math
import torch
import torch.nn.functional as F
from diffusers.utils import (
    BACKENDS_MAPPING,
    deprecate,
    logging
)
import inspect
from typing import Callable, List, Optional, Tuple, Union
from .block_sparse_flash_attention import block_sparse_flash_attention_forward
logger = logging.get_logger(__name__) 

def register_forward(
    model, 
    filter_name: str = 'Attention', 
    keep_shape: bool = True,
    sa_kward: dict = None,
    ca_kward: dict = None,
    ap: str = 'default',
    **kwargs,
):
    """
    A customized forward function for cross attention layer. 
    Detailed information in https://github.com/HaozheLiu-ST/T-GATE

    Args:
        model (`torch.nn.Module`):
            A diffusion model contains cross attention layers.
        filter_name (`str`):
            The name to filter the selected layer.
        keep_shape (`bool`):
            Whether or not to remain the shape of hidden features
        sa_kward: (`dict`):
            A kwargs dictionary to pass along to the self attention for caching and reusing. 
        ca_kward: (`dict`):
            A kwargs dictionary to pass along to the cross attention for caching and reusing. 

    Returns:
        count (`int`): The number of the cross attention layers used in the given model.
    """

    count = 0
    def warp_custom(
        self: torch.nn.Module,
        keep_shape:bool = True,
        ca_kward: dict = None,
        sa_kward: dict = None,
        **kwargs,
    ):
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            keep_shape = keep_shape,
            ca_cache = ca_kward['cache'],
            sa_cache = sa_kward['cache'],
            ca_reuse = ca_kward['reuse'],
            sa_reuse = sa_kward['reuse'],
            **cross_attention_kwargs,
        ) -> torch.Tensor:
            r"""
            The forward method of the `Attention` class.

            Args:
                hidden_states (`torch.Tensor`):
                    The hidden states of the query.
                encoder_hidden_states (`torch.Tensor`, *optional*):
                    The hidden states of the encoder.
                attention_mask (`torch.Tensor`, *optional*):
                    The attention mask to use. If `None`, no mask is applied.
                **cross_attention_kwargs:
                    Additional keyword arguments to pass along to the cross attention.

            Returns:
                `torch.Tensor`: The output of the attention layer.
            """
            # The `Attention` class can call different attention processors / attention functions
            # here we simply pass along all tensors to the selected processor class
            # For standard processors that are defined here, `**cross_attention_kwargs` is empty

            if not hasattr(self,'cache'):
                self.cache = None
            attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
            unused_kwargs = [k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters]
            
            if len(unused_kwargs) > 0:
                logger.warning(
                    f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
                )
            
            cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}
            hidden_states, cache =  tgate_processor(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                keep_shape = keep_shape,
                cache = self.cache,
                ca_cache = ca_cache,
                sa_cache = sa_cache,
                ca_reuse = ca_reuse,
                sa_reuse = sa_reuse,
                **cross_attention_kwargs,
            )
            if cache != None:
                self.cache = cache
            return hidden_states
        return forward
    
    def warp_custom_il(
        self: torch.nn.Module,
        keep_shape:bool = True,
        ca_kward: dict = None,
        sa_kward: dict = None,
        **kwargs,
    ):
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            query_rotary_emb: Optional[torch.Tensor] = None,
            key_rotary_emb: Optional[torch.Tensor] = None,
            base_sequence_length: Optional[int] = None,
            keep_shape = keep_shape,
            ca_cache = ca_kward['cache'],
            sa_cache = sa_kward['cache'],
            ca_reuse = ca_kward['reuse'],
            sa_reuse = sa_kward['reuse'],
            **cross_attention_kwargs,
        ) -> torch.Tensor:
            r"""
            The forward method of the `Attention` class.

            Args:
                hidden_states (`torch.Tensor`):
                    The hidden states of the query.
                encoder_hidden_states (`torch.Tensor`, *optional*):
                    The hidden states of the encoder.
                attention_mask (`torch.Tensor`, *optional*):
                    The attention mask to use. If `None`, no mask is applied.
                **cross_attention_kwargs:
                    Additional keyword arguments to pass along to the cross attention.

            Returns:
                `torch.Tensor`: The output of the attention layer.
            """
            # The `Attention` class can call different attention processors / attention functions
            # here we simply pass along all tensors to the selected processor class
            # For standard processors that are defined here, `**cross_attention_kwargs` is empty

            if not hasattr(self,'cache'):
                self.cache = None
            attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
            unused_kwargs = [k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters]
            
            if len(unused_kwargs) > 0:
                logger.warning(
                    f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
                )
            
            cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}
            hidden_states, cache =  tgate_processor_il(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                query_rotary_emb=query_rotary_emb,
                key_rotary_emb=key_rotary_emb,
                base_sequence_length=base_sequence_length,
                keep_shape = keep_shape,
                cache = self.cache,
                ca_cache = ca_cache,
                sa_cache = sa_cache,
                ca_reuse = ca_reuse,
                sa_reuse = sa_reuse,
                **cross_attention_kwargs,
            )
            if cache != None:
                self.cache = cache
            return hidden_states
        return forward

    def warp_custom_ja(
        self: torch.nn.Module,
        keep_shape:bool = True,
        ca_kward: dict = None,
        sa_kward: dict = None,
        **kwargs,
    ):
        def forward(
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            keep_shape = keep_shape,
            ca_cache = ca_kward['cache'],
            sa_cache = sa_kward['cache'],
            ca_reuse = ca_kward['reuse'],
            sa_reuse = sa_kward['reuse'],
            **cross_attention_kwargs,
        ) -> torch.Tensor:
            r"""
            The forward method of the `Attention` class.

            Args:
                hidden_states (`torch.Tensor`):
                    The hidden states of the query.
                encoder_hidden_states (`torch.Tensor`, *optional*):
                    The hidden states of the encoder.
                attention_mask (`torch.Tensor`, *optional*):
                    The attention mask to use. If `None`, no mask is applied.
                **cross_attention_kwargs:
                    Additional keyword arguments to pass along to the cross attention.

            Returns:
                `torch.Tensor`: The output of the attention layer.
            """
            # The `Attention` class can call different attention processors / attention functions
            # here we simply pass along all tensors to the selected processor class
            # For standard processors that are defined here, `**cross_attention_kwargs` is empty

            if not hasattr(self,'cache'):
                self.cache = None
            attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
            unused_kwargs = [k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters]
            
            if len(unused_kwargs) > 0:
                logger.warning(
                    f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
                )
            
            cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}
            hidden_states, ctx_hidden_states, cache =  tgate_processor_ja(
                self,
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                keep_shape = keep_shape,
                cache = self.cache,
                ca_cache = ca_cache,
                sa_cache = sa_cache,
                ca_reuse = ca_reuse,
                sa_reuse = sa_reuse,
                **cross_attention_kwargs,
            )
            if cache != None:
                self.cache = cache
            return hidden_states, ctx_hidden_states
        return forward




    def register_recr(
        net: torch.nn.Module, 
        count: int = None, 
        keep_shape:bool = True, 
        ca_kward:dict = None,
        sa_kward:dict = None,
        ap = 'default',
    ):
        if net.__class__.__name__ == filter_name:
            if ap == 'ja':
                net.forward = warp_custom_ja(net, keep_shape = keep_shape, ca_kward = ca_kward,sa_kward = sa_kward)
            elif ap == 'il':
                net.forward = warp_custom_il(net, keep_shape = keep_shape, ca_kward = ca_kward,sa_kward = sa_kward)
            else:
                net.forward = warp_custom(net, keep_shape = keep_shape, ca_kward = ca_kward,sa_kward = sa_kward)
            return count + 1
        elif hasattr(net, 'children'):
            for net_child in net.children():
                count = register_recr(net_child, count, keep_shape = keep_shape, ca_kward = ca_kward,sa_kward = sa_kward, ap = ap)
        return count
    count = register_recr(model, count, keep_shape = keep_shape, ca_kward = ca_kward,sa_kward = sa_kward, ap = ap)
    return count


def tgate_processor(
    attn = None,
    hidden_states = None,
    encoder_hidden_states = None,
    attention_mask  = None,
    temb  = None,
    cache = None,
    keep_shape = True,
    ca_cache = False,
    sa_cache = False,
    ca_reuse = False,
    sa_reuse = False,
    *args,
    **kwargs,
) -> torch.FloatTensor:

    r"""
    A customized forward function of the `AttnProcessor2_0` class.

    Args:
        hidden_states (`torch.Tensor`):
            The hidden states of the query.
        encoder_hidden_states (`torch.Tensor`, *optional*):
            The hidden states of the encoder.
        attention_mask (`torch.Tensor`, *optional*):
            The attention mask to use. If `None`, no mask is applied.
        **cross_attention_kwargs:
            Additional keyword arguments to pass along to the cross attention.

    Returns:
        `torch.Tensor`: The output of the attention layer.
    """
    # print('Using T-GATE processor for cross attention')
    if not hasattr(F, "scaled_dot_product_attention"):
        raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    if len(args) > 0 or kwargs.get("scale", None) is not None:
        deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
        deprecate("scale", "1.0.0", deprecation_message)

    residual = hidden_states

    cross_attn = encoder_hidden_states is not None
    self_attn =  encoder_hidden_states is None

    input_ndim = hidden_states.ndim

    # if the attention is cross-attention or self-attention
    if cross_attn and ca_reuse and cache != None:
        hidden_states = cache
        print('reuse cross attention')
    elif self_attn and sa_reuse and cache != None:
        hidden_states = cache
        print('reuse self attention')
    else:

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if (cross_attn and ca_cache) or (self_attn and sa_cache):
            if keep_shape:
                cache = hidden_states
            else:
                hidden_uncond, hidden_pred_text = hidden_states.chunk(2)
                cache = (hidden_uncond + hidden_pred_text ) / 2
        else:
            cache = None

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    if attn.residual_connection:
        hidden_states = hidden_states + residual

    hidden_states = hidden_states / attn.rescale_output_factor

    return hidden_states, cache


def tgate_processor_ja(
    attn = None,
    hidden_states: torch.FloatTensor = None,
    encoder_hidden_states: torch.FloatTensor = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    cache = None,
    keep_shape = True,
    ca_cache = False,
    sa_cache = False,
    ca_reuse = False,
    sa_reuse = False,
    *args,
    **kwargs,
) -> torch.FloatTensor:

    r"""
    A customized forward function of the `AttnProcessor2_0` class.

    Args:
        hidden_states (`torch.Tensor`):
            The hidden states of the query.
        encoder_hidden_states (`torch.Tensor`, *optional*):
            The hidden states of the encoder.
        attention_mask (`torch.Tensor`, *optional*):
            The attention mask to use. If `None`, no mask is applied.
        **cross_attention_kwargs:
            Additional keyword arguments to pass along to the cross attention.

    Returns:
        `torch.Tensor`: The output of the attention layer.
    """
    # print('Using T-GATE processor for joint attention')
    residual = hidden_states

    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    context_input_ndim = encoder_hidden_states.ndim
    if context_input_ndim == 4:
        batch_size, channel, height, width = encoder_hidden_states.shape
        encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

    batch_size = encoder_hidden_states.shape[0]

    # `sample` projections.
    if sa_reuse and cache != None:
        query=cache['query']
        key=cache['key']
        value=cache['value']
        # print('reuse self attention')
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        if sa_cache:
            cache = cache or {}
            cache.update({
                'query': query,
                'key': key,
                'value': value,
            })
        # print('cache self attention')

    # `context` projections.
    if ca_reuse and cache != None:
        encoder_hidden_states_query_proj=cache['encoder_hidden_states_query_proj']
        encoder_hidden_states_key_proj=cache['encoder_hidden_states_key_proj']
        encoder_hidden_states_value_proj=cache['encoder_hidden_states_value_proj']
        # print('reuse cross attention')
    else:
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        # print('encoder_hidden_states_query_proj', encoder_hidden_states_query_proj.shape)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        if ca_cache:
            cache = cache or {}
            cache.update({
                'encoder_hidden_states_query_proj': encoder_hidden_states_query_proj,
                'encoder_hidden_states_key_proj': encoder_hidden_states_key_proj,
                'encoder_hidden_states_value_proj': encoder_hidden_states_value_proj,
            })
        # print('cache cross attention')

    # attention
    # print('query', query.shape)
    len_q = query.shape[1]
    len_ctx = encoder_hidden_states_query_proj.shape[1]
    query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
    key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
    value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    mask = torch.zeros((2, 24, query.shape[2], query.shape[2]), device=query.device, dtype=torch.bool)
    if (sa_reuse and cache != None) or (ca_reuse and cache != None):
        mask[:, :len_q] = True
    if ca_reuse and cache != None:
        mask[:, len_q:] = True
    # if sa_reuse and cache != None:
    #     block_mask = torch.rand((2, 24, 42, 78), device="cuda") > (len_q / (len_q + len_ctx))
    # elif  ca_reuse and cache != None:
    #     block_mask = torch.rand((2, 24, 42, 78), device="cuda") > (len_ctx / (len_q + len_ctx))
    # else:
    #     block_mask = torch.ones((2, 24, 42, 72), device="cuda")
    # block_mask = torch.rand((2, 24, 35, 70), device="cuda") > 0.5
    # print( mask.shape)

    # hidden_states = hidden_states = F.scaled_dot_product_attention(
    #     query, key, value, dropout_p=0.0, is_causal=False, attn_mask=mask
    # )
    hidden_states = hidden_states = F.scaled_dot_product_attention(
        query, key, value, dropout_p=0.0, is_causal=False
    )
    # hidden_states = block_sparse_flash_attention_forward(
    #     query, key, value, block_mask
    # )
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)

    # Split the attention outputs.
    hidden_states, encoder_hidden_states = (
        hidden_states[:, : residual.shape[1]],
        hidden_states[:, residual.shape[1] :],
    )

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)
    if not attn.context_pre_only:
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
    if context_input_ndim == 4:
        encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

    return hidden_states, encoder_hidden_states, cache

def tgate_processor_il(
    attn,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    query_rotary_emb: Optional[torch.Tensor] = None,
    key_rotary_emb: Optional[torch.Tensor] = None,
    base_sequence_length: Optional[int] = None,
    cache = None,
    keep_shape = True,
    ca_cache = False,
    sa_cache = False,
    ca_reuse = False,
    sa_reuse = False,
) -> torch.Tensor:
    from diffusers.models.embeddings import apply_rotary_emb

    self_attn = hidden_states.shape == encoder_hidden_states.shape
    cross_attn = not self_attn

    if cross_attn and ca_reuse and cache != None:
        hidden_states = cache
        # print('reuse cross attention')
    elif self_attn and sa_reuse and cache != None:
        hidden_states = cache
        # print('reuse self attention')
    else:

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        # Get Query-Key-Value Pair
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query_dim = query.shape[-1]
        inner_dim = key.shape[-1]
        head_dim = query_dim // attn.heads
        dtype = query.dtype

        # Get key-value heads
        kv_heads = inner_dim // head_dim

        # Apply Query-Key Norm if needed
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.view(batch_size, -1, attn.heads, head_dim)

        key = key.view(batch_size, -1, kv_heads, head_dim)
        value = value.view(batch_size, -1, kv_heads, head_dim)

        # Apply RoPE if needed
        if query_rotary_emb is not None:
            query = apply_rotary_emb(query, query_rotary_emb, use_real=False)
        if key_rotary_emb is not None:
            key = apply_rotary_emb(key, key_rotary_emb, use_real=False)

        query, key = query.to(dtype), key.to(dtype)

        # Apply proportional attention if true
        if key_rotary_emb is None:
            softmax_scale = None
        else:
            if base_sequence_length is not None:
                softmax_scale = math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale
            else:
                softmax_scale = attn.scale

        # perform Grouped-qurey Attention (GQA)
        n_rep = attn.heads // kv_heads
        if n_rep >= 1:
            key = key.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)
            value = value.unsqueeze(3).repeat(1, 1, 1, n_rep, 1).flatten(2, 3)

        # scaled_dot_product_attention expects attention_mask shape to be
        # (batch, heads, source_length, target_length)
        attention_mask = attention_mask.bool().view(batch_size, 1, 1, -1)
        attention_mask = attention_mask.expand(-1, attn.heads, sequence_length, -1)

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        # print(query.shape, key.shape, value.shape, attention_mask.shape)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, scale=softmax_scale
        )
        hidden_states = hidden_states.transpose(1, 2).to(dtype)

        if (cross_attn and ca_cache) or (self_attn and sa_cache):
            if keep_shape:
                cache = hidden_states
            else:
                hidden_uncond, hidden_pred_text = hidden_states.chunk(2)
                cache = (hidden_uncond + hidden_pred_text ) / 2
        else:
            cache = None

    return hidden_states, cache

def tgate_scheduler(
    cur_step: int = None, 
    gate_step: int = 10, 
    sp_interval: int = 5,
    fi_interval: int = 1,
    warm_up: int = 2,
):
    r"""
    The T-GATE scheduler function 

    Args:
        cur_step (`int`):
            The current time step. 
        gate_step (`int` defaults to 10): 
            The time step to stop calculating the cross attention.
        sp_interval (`int` defaults to 5): 
            The time-step interval to cache self attention before gate_step (Semantics-Planning Phase).
        fi_interval (`int` defaults to 1): 
            The time-step interval to cache self attention after gate_step (Fidelity-Improving Phase).
        warm_up (`int` defaults to 2): 
            The time step to warm up the model inference.

    Returns:
        ca_kward: (`dict`):
            A kwargs dictionary to pass along to the cross attention for caching and reusing. 
        sa_kward: (`dict`):
            A kwargs dictionary to pass along to the self attention for caching and reusing. 
        keep_shape (`bool`):
            Whether or not to remain the shape of hidden features
    """
    if cur_step < gate_step-1:
        # Semantics-Planning Stage
        ca_kwards = {
            'cache': False,
            'reuse': False,
        }
        if cur_step < warm_up:
            sa_kwards = {
                'cache': False,
                'reuse': False,
            }
        elif cur_step == warm_up:
            sa_kwards = {
                'cache': True,
                'reuse': False,
            }   
        else:
            if cur_step % sp_interval == 0:
                sa_kwards = {
                    'cache': True,
                    'reuse': False,
                }
            else:
                sa_kwards = {
                    'cache': False,
                    'reuse': True,
                }
        keep_shape = True
    
    elif cur_step == gate_step-1:
        ca_kwards = {
            'cache': True,
            'reuse': False,
        }
        sa_kwards = {
            'cache':True,
            'reuse':False
        }
        keep_shape = False
    else:
        # Fidelity-Improving Stage
        ca_kwards = {
            'cache': False,
            'reuse': True,
        }
        if cur_step % fi_interval == 0:
            sa_kwards = {
                'cache':True,
                'reuse':False
            }
        else:
            sa_kwards = {
                'cache':False,
                'reuse':True
            }
        keep_shape = True
    return ca_kwards, sa_kwards, keep_shape
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import os
from scipy.sparse import load_npz
from torch import Tensor
from torch.nn.utils import parametrize

import logging
import numpy as np

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.transformer import (
    TransformerConfig,
    TransformerEncoderBase,
)

from .transformer_id.transformer_decoder import TransformerDecoderBase

from fairseq.modules import FairseqDropout

logger = logging.getLogger(__name__)


class TransformerModelBase(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        gnn_dropout = FairseqDropout(cfg.gnn_dropout, module_name="gnn_dropout")
        print("gnn dropout ratio: {}".format(cfg.gnn_dropout))

        # build graphmerger
        if torch.cuda.is_available():
            # Get the current device
            device = torch.cuda.current_device()
            print("Current Device:", device)
        else:
            # If CUDA is not available, use the CPU
            device = torch.device("cpu")
            print("Current Device: CPU")

        if cfg.fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32

        if cfg.merger == 'naive':
            print('current merger: naive merger')
            Merger = NaiveMerge
        else:
            print('current merger: graph merger')
            Merger = GraphMerge


        if cfg.shared_graph:
            print("share graph")
            encoder_graph_merger = Merger(cfg.graph_path, cfg.encoder.embed_dim, device, dtype, gnn_dropout, cfg.hop_num)
            decoder_graph_merger = encoder_graph_merger
        else:
            print("not share graph")
            encoder_graph_merger = Merger(cfg.graph_path, cfg.encoder.embed_dim, device, dtype, gnn_dropout, cfg.hop_num)
            decoder_graph_merger = Merger(cfg.graph_path, cfg.encoder.embed_dim, device, dtype, gnn_dropout, cfg.hop_num)

        if cfg.share_all_embeddings:
            print("share all embeddings")
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                    cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens, graph_merger = cls.build_graph_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, gnn_dropout, cfg.graph_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        elif cfg.merge_src_tgt_embed:
            print("merge src&tgt embeddings")
            logger.info(f"source dict size: {len(src_dict)}")
            logger.info(f"target dict size: {len(tgt_dict)}")
            src_dict.update(tgt_dict)
            task.src_dict = src_dict
            task.tgt_dict = src_dict
            logger.info(f"merged dict size: {len(src_dict)}")
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            print("else embeddings")
            encoder_embed_tokens = cls.build_graph_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, encoder_graph_merger
            )
            decoder_embed_tokens = cls.build_graph_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, decoder_graph_merger
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens, decoder_graph_merger)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_graph_embedding(cls, cfg, dictionary, embedding_dim, graph_merger, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = MultiHopGraphEmbedding(num_embeddings, embedding_dim, padding_idx, graph_merger)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens, graph_merger):
        return TransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            graph_merger=graph_merger,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            return_all_hiddens: bool = True,
            features_only: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class GraphMerge(nn.Module):
    def __init__(self, matrix_path, dim, device, type, gnn_dropout, hop_num):
        super().__init__()
        self.graph_matrix = None
        self.hop_num = hop_num
        self.dtype = type

        load_sparse_np = load_npz(matrix_path)
        indices = torch.from_numpy(np.vstack((load_sparse_np.row, load_sparse_np.col)).astype(np.int64))
        values = torch.from_numpy(load_sparse_np.data.astype(np.float32))
        size = torch.Size(load_sparse_np.shape)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size)

        coo_tensor = sparse_tensor.coalesce()
        csr_tensor = coo_tensor.to_sparse_csr()
        csr_tensor = csr_tensor.to(type)
        
        self.graph_matrix = csr_tensor.cuda().to(type)

        # init linear_layers
        self.neighbor_layers = nn.ModuleList([])
        self.neighbor_layers.extend([nn.Linear(dim, dim, bias=True, device=device, dtype=type).to(device)
                                     for _ in range(self.hop_num)])
        self.self_layers = nn.ModuleList([])
        self.self_layers.extend([nn.Linear(dim, dim, bias=True, device=device, dtype=type).to(device)
                                 for _ in range(self.hop_num)])
        # self.merge_layers = nn.ModuleList([])
        # self.merge_layers.extend([nn.Linear(dim, dim, bias=True, device=device, dtype=type).to(device)
        #                           for _ in range(self.hop_num)])

        # init nonlinear_func
        self.nonlinear_func = nn.ELU()
        # self.nonlinear_func = nn.Tanh()

        # init dropout
        self.gnn_dropout = gnn_dropout

    def forward(self, m):
        m = m.cuda()

        # multi-hop message passing
        H_merge = m.to(self.dtype)

        for i in range(self.hop_num):
            H_merge = self.gnn_dropout(H_merge)
            H_neighbor = torch.matmul(self.graph_matrix, H_merge)  # [V, Dim]
            H_neighbor = self.neighbor_layers[i](H_neighbor)       # [V, Dim]
            H_self = self.self_layers[i](H_merge)                  # [V, Dim]
            H_merge = H_neighbor + H_self
            # H_merge = self.merge_layers[i](H_merge)
            if i != self.hop_num - 1:
                H_merge = self.nonlinear_func(H_merge)             # [V, Dim]

        H_merge = H_merge.to(m.dtype)
        return H_merge


class NaiveMerge(nn.Module):
    def __init__(self, matrix_path, dim, device, type, gnn_dropout, hop_num):
        super().__init__()
        self.graph_matrix = None
        self.hop_num = hop_num
        self.dtype = type

        load_sparse_np = load_npz(matrix_path)
        indices = torch.from_numpy(np.vstack((load_sparse_np.row, load_sparse_np.col)).astype(np.int64))
        values = torch.from_numpy(load_sparse_np.data.astype(np.float32))
        size = torch.Size(load_sparse_np.shape)
        sparse_tensor = torch.sparse_coo_tensor(indices, values, size)

        coo_tensor = sparse_tensor.coalesce()
        csr_tensor = coo_tensor.to_sparse_csr()
        csr_tensor = csr_tensor.to(type)

        self.graph_matrix = csr_tensor.cuda().to(type)

        # init dropout
        self.gnn_dropout = gnn_dropout

    def forward(self, m):
        m = m.cuda()

        # multi-hop message passing
        H_merge = m.to(self.dtype)
        H_merge = torch.matmul(self.graph_matrix, H_merge)
        H_merge = H_merge.to(m.dtype)
        return H_merge


def MultiHopGraphEmbedding(num_embeddings, embedding_dim, padding_idx, graph_merger):
    # m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx, device="cuda")
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)

    parametrize.register_parametrization(m, "weight", graph_merger)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


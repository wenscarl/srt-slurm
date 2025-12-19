# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
TensorRT-LLM backend configuration (placeholder).

Implements BackendProtocol for TRT-LLM inference serving.
"""

from dataclasses import field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Type,
)

from marshmallow import Schema
from marshmallow_dataclass import dataclass

if TYPE_CHECKING:
    from srtctl.core.endpoints import Endpoint, Process
    from srtctl.core.runtime import RuntimeContext


@dataclass(frozen=True)
class TRTLLMBackendConfig:
    """TensorRT-LLM backend configuration - implements BackendProtocol.

    Placeholder for future TRT-LLM support.

    Example YAML:
        backend:
          type: trtllm
          engine_path: /path/to/engine
          max_batch_size: 64
    """

    type: Literal["trtllm"] = "trtllm"

    # TRT-LLM specific settings
    engine_path: Optional[str] = None
    max_batch_size: int = 64
    max_input_len: int = 2048
    max_output_len: int = 2048

    # Disaggregated mode support
    prefill_environment: Dict[str, str] = field(default_factory=dict)
    decode_environment: Dict[str, str] = field(default_factory=dict)
    aggregated_environment: Dict[str, str] = field(default_factory=dict)

    Schema: ClassVar[Type[Schema]] = Schema

    # =========================================================================
    # BackendProtocol Implementation (stubs)
    # =========================================================================

    def get_config_for_mode(self, mode: str) -> Dict[str, Any]:
        """Get config for TRT-LLM mode."""
        return {
            "engine_path": self.engine_path,
            "max_batch_size": self.max_batch_size,
            "max_input_len": self.max_input_len,
            "max_output_len": self.max_output_len,
        }

    def get_environment_for_mode(self, mode: str) -> Dict[str, str]:
        """Get environment variables for a worker mode."""
        if mode == "prefill":
            return dict(self.prefill_environment)
        elif mode == "decode":
            return dict(self.decode_environment)
        elif mode == "agg":
            return dict(self.aggregated_environment)
        return {}

    def allocate_endpoints(
        self,
        num_prefill: int,
        num_decode: int,
        num_agg: int,
        gpus_per_prefill: int,
        gpus_per_decode: int,
        gpus_per_agg: int,
        gpus_per_node: int,
        available_nodes: Sequence[str],
    ) -> List["Endpoint"]:
        """Allocate endpoints for TRT-LLM."""
        from srtctl.core.endpoints import allocate_endpoints

        return allocate_endpoints(
            num_prefill=num_prefill,
            num_decode=num_decode,
            num_agg=num_agg,
            gpus_per_prefill=gpus_per_prefill,
            gpus_per_decode=gpus_per_decode,
            gpus_per_agg=gpus_per_agg,
            gpus_per_node=gpus_per_node,
            available_nodes=available_nodes,
        )

    def endpoints_to_processes(
        self,
        endpoints: List["Endpoint"],
        base_port: int = 8081,
    ) -> List["Process"]:
        """Convert endpoints to processes."""
        from srtctl.core.endpoints import endpoints_to_processes

        return endpoints_to_processes(endpoints, base_port)

    def build_worker_command(
        self,
        process: "Process",
        endpoint_processes: List["Process"],
        runtime: "RuntimeContext",
        use_sglang_router: bool = False,
        dump_config_path: Optional[Path] = None,
    ) -> List[str]:
        """Build TRT-LLM server command."""
        from srtctl.core.runtime import get_hostname_ip

        mode = process.endpoint_mode

        # Determine if multi-node
        endpoint_nodes = list(dict.fromkeys(p.node for p in endpoint_processes))
        is_multi_node = len(endpoint_nodes) > 1

        cmd = [
            "python3",
            "-m",
            "dynamo.trtllm",
            "--model-path",
            str(runtime.model_path),
            "--host",
            "0.0.0.0",
        ]

        # Add mode flag for disaggregated
        if mode != "agg":
            cmd.extend(["--disaggregation-mode", mode])

        # Engine path
        if self.engine_path:
            cmd.extend(["--engine-path", self.engine_path])

        # Multi-node coordination
        if is_multi_node:
            leader_ip = get_hostname_ip(endpoint_nodes[0])
            node_rank = endpoint_nodes.index(process.node)
            cmd.extend(
                [
                    "--dist-init-addr",
                    f"{leader_ip}:29500",
                    "--nnodes",
                    str(len(endpoint_nodes)),
                    "--node-rank",
                    str(node_rank),
                ]
            )

        return cmd


# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backend configuration dataclasses.

Each backend has its own frozen dataclass that implements the BackendProtocol.
"""

from .base import BackendProtocol, BackendType
from .sglang import SGLangBackendConfig, SGLangConfig
from .trtllm import TRTLLMBackendConfig
from .vllm import VLLMBackendConfig

# Union type for all backend configs
BackendConfig = SGLangBackendConfig | VLLMBackendConfig | TRTLLMBackendConfig

__all__ = [
    # Base types
    "BackendProtocol",
    "BackendType",
    "BackendConfig",
    # SGLang
    "SGLangBackendConfig",
    "SGLangConfig",
    # vLLM
    "VLLMBackendConfig",
    # TRT-LLM
    "TRTLLMBackendConfig",
]


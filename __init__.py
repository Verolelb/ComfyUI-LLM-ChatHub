"""LLM ChatHub for ComfyUI.

A chat client (sidebar panel + nodes) for LM Studio, Ollama and Ollama Cloud with
persistent conversation history, editable prompts, images and optional document RAG.
"""

import logging

from .llm_chathub import nodes, routes

LOGGER = logging.getLogger("llm_chathub")

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = nodes.NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = nodes.NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

try:
    routes.register_routes()
except Exception as error:  # pragma: no cover - nodes still load if the API fails
    LOGGER.warning("llm_chathub: impossible d'enregistrer les routes API (%s)", error)

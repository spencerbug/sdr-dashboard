#!/usr/bin/env python3
"""Entry point for the web-based YCB viewer backend."""

import os
from typing import Callable, Optional

import uvicorn


def main() -> None:
    host = os.environ.get("YCB_VIEWER_HOST", "0.0.0.0")
    port = int(os.environ.get("YCB_VIEWER_PORT", "8000"))
    reload_env = os.environ.get("YCB_VIEWER_RELOAD", "0")
    reload = reload_env == "1"
    if reload:
        print("WARNING: reload=True can break asyncio locks and cause frame loop errors. Use only for frontend dev!")
    log_level = os.environ.get("YCB_VIEWER_LOG_LEVEL", "info")

    config = uvicorn.Config(
        "backend.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        timeout_graceful_shutdown=15,
    )
    server = uvicorn.Server(config)

    handler: Optional[Callable[[], None]] = None
    try:
        config.load()
        from backend.server import register_shutdown_handler

        def trigger_shutdown() -> None:
            server.should_exit = True

        register_shutdown_handler(trigger_shutdown)
        handler = trigger_shutdown
    except Exception:
        handler = None

    try:
        server.run()
    except KeyboardInterrupt:
        server.should_exit = True
        if handler is not None:
            handler()


if __name__ == "__main__":
    main()

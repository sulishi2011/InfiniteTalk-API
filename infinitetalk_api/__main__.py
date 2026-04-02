from __future__ import annotations

import uvicorn

from .config import ServiceConfig


def main() -> None:
    config = ServiceConfig.from_env()
    uvicorn.run(
        "infinitetalk_api.main:app",
        host=config.host,
        port=config.port,
        reload=False,
    )


if __name__ == "__main__":
    main()


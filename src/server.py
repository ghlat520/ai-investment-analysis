"""
uvicorn 入口
"""

import uvicorn

from src.api.app import create_app

app = create_app()


def main(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(
        "src.server:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()


"""src/logging_setup.py
集中式日志配置。

约定（标准库的 library-vs-application 分工）：
- 只有入口（各文件的 __main__）调用 setup_logging()
- 被 import 的模块只写 logger = logging.getLogger(__name__)，绝不碰 handler

这样 extract.py 被 evaluate.py import 时不会重复装 handler，
pytest 里没人调用 setup_logging 时也只会走 lastResort（WARNING 以上到 stderr）。
"""
import logging
import os
import sys
from pathlib import Path

project_root_dir = Path(__file__).resolve().parent.parent

CONSOLE_FORMAT = "%(levelname)-7s %(name)-10s %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)-7s %(name)-10s %(funcName)s:%(lineno)d  %(message)s"


def setup_logging(level: str | None = None, log_file: str | None = "out/run.log") -> logging.Logger:
    """装好 root logger。level 可用 LOG_LEVEL 环境变量覆盖，默认 INFO。

    控制台走 stderr，文件永远记 DEBUG 全量——跑完一轮 eval 之后还能回去查
    某一行为什么重试了三次。
    """
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    if root.handlers:                      # 幂等：重复调用不叠加 handler
        return root

    root.setLevel(logging.DEBUG)           # 阈值交给各 handler 决定

    console = logging.StreamHandler(sys.stderr)   # 不污染 stdout：报告还能重定向
    console.setLevel(level)
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT))
    root.addHandler(console)

    if log_file:
        path = project_root_dir / log_file
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        root.addHandler(file_handler)

    # 第三方库的噪声：httpx 会把每个请求都打成 INFO
    for noisy in ("httpx", "httpx2", "httpcore", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root

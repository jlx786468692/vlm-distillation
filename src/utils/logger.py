"""
Logger Setup Module
===================

Provides logging configuration and utilities for the distillation pipeline.
"""

import os
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional
from datetime import datetime


# ANSI 颜色代码
class LogColors:
    """ANSI color codes for terminal output."""
    # 日志级别颜色（整行统一颜色）
    RED = '\033[91m'      # ERROR - 红色（整行）
    YELLOW = '\033[93m'   # DEBUG - 黄色（整行）
    ORANGE = '\033[33m'   # WARNING - 橙色（整行）
    WHITE = '\033[97m'    # INFO/其他 - 白色（整行）
    GREEN = '\033[92m'    # SUCCESS - 绿色（整行）
    CYAN = '\033[96m'     # 其他颜色（整行）

    # 控制代码
    RESET = '\033[0m'     # 重置颜色
    BOLD = '\033[1m'      # 加粗


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output - 整行根据日志等级统一颜色."""

    # 日志级别对应的颜色（整行颜色）
    LEVEL_COLORS = {
        logging.DEBUG: LogColors.CYAN,      # DEBUG - 青色（整行）
        logging.INFO: LogColors.WHITE,     # INFO - 白色（整行）
        logging.WARNING: LogColors.YELLOW,  # WARNING - 黄色（整行）
        logging.ERROR: LogColors.RED,       # ERROR - 红色（整行）
        logging.CRITICAL: LogColors.BOLD + LogColors.RED,  # CRITICAL - 粗体红色（整行）
    }

    def format(self, record):
        """格式化日志记录，整行统一颜色"""
        # 获取对应级别的颜色（整行颜色）
        line_color = self.LEVEL_COLORS.get(record.levelno, LogColors.WHITE)
        reset = LogColors.RESET

        # 先调用父类的 format 方法获取格式化的字符串
        formatted = super().format(record)

        # 给整行添加颜色（包括时间戳、logger名称、模块名、消息等所有部分）
        colored_formatted = f"{line_color}{formatted}{reset}"

        return colored_formatted


def setup_logger(
    name: str = "vlm_distillation",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True,
    format_string: Optional[str] = None,
    use_colors: bool = True
) -> logging.Logger:
    """
    Set up a logger with configurable handlers.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, no file logging.
        console_output: Whether to output to console
        format_string: Custom format string. If None, uses default.
        use_colors: Whether to use colored output for console.

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers.clear()

    # Default format - 添加详细位置信息
    if format_string is None:
        format_string = "[%(asctime)s] [%(levelname)s] [%(name)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s"

    # Console handler - 使用彩色格式
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        if use_colors:
            console_handler.setFormatter(ColoredFormatter(format_string, datefmt="%Y-%m-%d %H:%M:%S"))
        else:
            console_handler.setFormatter(logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S"))
        
        logger.addHandler(console_handler)

    # File handler - 不使用颜色（文件中不需要ANSI代码）
    if log_file:
        # Ensure log directory exists
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

    return logger


class DistillationLogger:
    """
    Specialized logger for distillation process with progress tracking.
    """

    def __init__(
        self,
        name: str = "distillation",
        log_dir: str = "./logs",
        level: str = "INFO"
    ):
        """
        Initialize DistillationLogger.

        Args:
            name: Logger name
            log_dir: Directory for log files
            level: Logging level
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(self.log_dir / f"{name}_{timestamp}.log")

        self.logger = setup_logger(
            name=name,
            level=level,
            log_file=log_file,
            console_output=True,
            use_colors=True
        )

        # Progress tracking
        self.start_time = None
        self.processed_count = 0
        self.total_count = 0

    def start_process(self, total_count: int, description: str = "Starting distillation") -> None:
        """
        Log process start and initialize progress tracking.

        Args:
            total_count: Total number of items to process
            description: Process description
        """
        self.start_time = datetime.now()
        self.total_count = total_count
        self.processed_count = 0

        self.logger.info(f"{description} - Total items: {total_count}")
        self.logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    def log_progress(
        self,
        current: int,
        message: str = "",
        log_interval: int = 10
    ) -> None:
        """
        Log processing progress.

        Args:
            current: Current processed count
            message: Additional message
            log_interval: Log every N items
        """
        self.processed_count = current

        if current % log_interval == 0 or current == self.total_count:
            percentage = (current / self.total_count) * 100 if self.total_count > 0 else 0
            elapsed = datetime.now() - self.start_time if self.start_time else 0

            self.logger.info(
                f"Progress: {current}/{self.total_count} ({percentage:.1f}%) | "
                f"Elapsed: {elapsed} | {message}"
            )

    def log_task_result(
        self,
        image_id: str,
        task: str,
        success: bool,
        details: Optional[str] = None
    ) -> None:
        """
        Log result for specific task on an image.

        Args:
            image_id: Image identifier
            task: Task name (vqa, captioning, detection)
            success: Whether task succeeded
            details: Additional details
        """
        status = "SUCCESS" if success else "FAILED"
        message = f"[{task}] Image {image_id}: {status}"
        if details:
            message += f" | {details}"

        if success:
            self.logger.info(message)
        else:
            self.logger.warning(message)

    def log_error(self, error: Exception, context: Optional[str] = None) -> None:
        """
        Log error with context and full traceback.

        Args:
            error: Exception object
            context: Error context
        """
        message = f"ERROR: {str(error)}"
        if context:
            message = f"{context} - {message}"

        # 使用 exc_info=True 显示完整 traceback
        self.logger.error(message, exc_info=True)
        
        # 额外打印完整的 traceback
        self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")

    def end_process(self, description: str = "Distillation completed") -> None:
        """
        Log process completion with statistics.

        Args:
            description: Completion description
        """
        if self.start_time:
            elapsed = datetime.now() - self.start_time
            self.logger.info(f"{description}")
            self.logger.info(f"Processed: {self.processed_count}/{self.total_count}")
            self.logger.info(f"Total time: {elapsed}")

            if self.processed_count > 0:
                avg_time = elapsed.total_seconds() / self.processed_count
                self.logger.info(f"Average time per item: {avg_time:.2f} seconds")

    def get_logger(self) -> logging.Logger:
        """Get underlying logger instance."""
        return self.logger


# Create default logger
_default_logger = None


def get_logger() -> logging.Logger:
    """
    Get default logger instance.

    Returns:
        Default logger
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger()
    return _default_logger
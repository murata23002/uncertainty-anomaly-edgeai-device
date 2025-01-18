import functools
import logging
import os
import time
from logging.handlers import RotatingFileHandler

from config_manager.config import LogConfigs


class CustomLogger:
    def __init__(self):
        self._log_conf = LogConfigs()
        self.logger = logging.getLogger("app_logger")
        self._initialize_logger()

    def _initialize_logger(self):
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(thread)d - %(levelname)s - %(module)s - %(funcName)s - %(message)s"
        )

        log_dir = os.path.dirname(self._log_conf.file)
        if log_dir != "" and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(
            filename=self._log_conf.file,
            encoding=self._log_conf.encoding,
        )
        # Use RotatingFileHandler for log rotation based on file size
        file_handler = RotatingFileHandler(
            filename=self._log_conf.file,
            maxBytes=5 * 1024 * 1024,  # 5 MB max per file
            backupCount=3,  # Keep up to 3 backup files
            encoding=self._log_conf.encoding,
        )
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.setLevel(getattr(logging, self._log_conf.level.upper()))

    def log_info_cls_properties(self, cls_obj):
        """
        オブジェクトのプロパティをログに出力する
        """
        self.logger.debug(f"Properties of {cls_obj.__class__.__name__}:")
        for key, value in vars(cls_obj).items():
            self.logger.info(f"{key}: {value}")

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        """
        例外情報は自動的に出力されるので msg には含めない
        """
        self.logger.exception(msg, *args, exc_info=True, **kwargs)

    def is_debug_enabled(self):
        return self.logger.isEnabledFor(logging.DEBUG)


custom_logger = CustomLogger()


def log_debug_method_execution(suppress_output=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not custom_logger.is_debug_enabled():
                return func(*args, **kwargs)

            method_name = func.__qualname__
            module_name = func.__module__
            custom_logger.debug(
                f"Starting execution of {method_name} in module: {module_name}"
            )

            if not suppress_output:
                custom_logger.debug(f"Input arguments: args={args}, kwargs={kwargs}")

            start_time = time.time()

            result = func(*args, **kwargs)

            execution_time = round(time.time() - start_time, 6)

            if not suppress_output:
                custom_logger.debug(f"Output result: {result}")

            custom_logger.debug(
                f"Finished execution of {method_name} in module: {module_name}. Execution time: {execution_time} seconds"
            )

            return result

        return wrapper

    return decorator

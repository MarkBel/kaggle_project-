import logging

_log_format = f"%(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - %(message)s"


class SimpleLogger():
    """
    create a logger that write logs into file also output it to stdout
    """

    def __init__(self, name: str, file_log: str) -> None:
        """
        constructor of logger
        Args:
            name (str): name of logging 
            file_log (str): output logs file
        """
        self.name = name
        self.file_log = file_log

    def get_file_handler(self) -> logging.FileHandler:
        """
        configures what the logger will write to the file

        Returns:
            logging.FileHandler: logger that write logs into file
        """
        file_handler = logging.FileHandler(self.file_log)
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(logging.Formatter(_log_format))
        return file_handler

    def get_stream_handler(self) -> logging.StreamHandler:
        """
        configures what the logger will write to the stdout
        Returns:
            logging.StreamHandler: logger that write logs to stdout
        """
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(logging.Formatter(_log_format))
        return stream_handler

    def get_logger(self) -> logging.Logger:
        """
        return already configured logger 

        Returns:
            logging.Logger: already configured logger 
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(logging.INFO)
        logger.addHandler(self.get_stream_handler())

        if self.file_log is not None:
            logger.addHandler(self.get_file_handler())
        return logger

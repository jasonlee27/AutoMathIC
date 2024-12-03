
import sys, os
import logging

from pathlib import Path

from .macros import Macros

# class MStreamHandler(logging.StreamHandler):
#     """Handler that controls the writing of the newline character"""
#     def __init__(self):
#         super().__init__()

#     def emit(self, record) -> None:
#         record.msg = record.msg[:-1]+self.terminator
#         return super().emit(record)
                            
class Logger:

    def __init__(self, logger_file: Path, logger_name=None, _format=None, mode='w'):
        self.logger_file = logger_file
        self.mode = 'w'
        if self.mode=='w' and os.path.exists(logger_file):
            os.remove(logger_file)
        # end if
        # self.logger_name = logger_name if logger_name is not None else __name__
        # self.terminator = ''
        # self._format = '%(message)s' if _format is None else _format
        # self.logger = self.get_logger()
        
    # def get_logger(self):
    #     # logging.basicConfig(
    #     #     filename=self.logger_file,
    #     #     filemode='w',
    #     #     level=logging.INFO,
    #     #     format=self._format
    #     # )
    #     logger = logging.getLogger(self.logger_name)
    #     fh = logging.FileHandler(self.logger_file)
    #     fh.setLevel(logging.INFO)
    #     fh_format = logging.Formatter(self._format)
    #     fh.setFormatter(fh_format)
    #     logger.addHandler(fh)
    #     h1 = MStreamHandler()
    #     h1.terminator = self.terminator
    #     logger.addHandler(h1)
    #     return logger
    
    def print(self, *argv, **kwargs):
        if any(argv):
            text = argv[0]
            end = '\n'
            if 'end' in kwargs.keys():
                end = kwargs['end']
            # end if
        else:
            text, end ='', '\n'
        # end if
        orig_stdout = sys.stdout
        with open(self.logger_file, 'a') as f:
            sys.stdout = f
            print(text, end=end)
            sys.stdout = orig_stdout
        # end with
        # self.logger.info(text+end)

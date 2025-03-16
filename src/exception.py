import sys
from src.logger import logging

def error_messege_detail(error, error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_messege = "Error occured in python script name [{0}] line number [{1}] error messege [{2}]".format(file_name, exc_tb.tb_lineno, str(error))

    return error_messege


class CustomException(Exception):
    def __init__(self, error_messege, error_detail:sys):
        super().__init__(error_messege)
        self.error_messege = error_messege_detail(error_messege, error_detail=error_detail)

    def __str__(self):
        return self.error_messege
    

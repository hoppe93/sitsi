# Implementation of basic exception class


class InverterException(Exception):
    

    def __init__(self, message):
        super(Exception, self).__init__(message)



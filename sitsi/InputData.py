"""
Base class for inversion input data.
"""

class InputData:
    
    def __init__(self): pass


    def get(self):
        """
        Returns the data of this object. Must be overridden.
        """
        raise InverterException("Method 'get()' not implemented by this class.")



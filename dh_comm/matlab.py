import matlab.engine
import numpy as np

# NOTE: to reuse matlab instance, call script using 
#       interactive mode
# >>>   run -i test_engine.py
try:
    eng.pwd()
except:
    print("restarting matlab engine")
    eng = matlab.engine.start_matlab()
    # add minimal matlab initialization code 
    # search paths, etc.
    mat_ws_path = '~/workspace/matlab'
    eng.cd( mat_ws_path )
    eng.startup(nargout=0) # run startup.m script
#    eng.test_python_call(nargout=0)

class MatlabProxy:

    def __init__(self):
        self.function_name = None
        self.eng = eng

    def __handler(self, *argv, **kwargs):
        '''
        supports arbitrary number of input arrays and one output array
        NOTE: input assumed to be numpy arrays
        NOTE: 1d-arrays are converted to (N,1)-matrices
              scalars are converted to (1,1)-matrices
        NOTE: array.tolist() recurse into all dimensions
              i.e., 2-dim array => 2-level nested list
        '''
        array_args = []
        for arg in argv: 
            if np.isscalar(arg):
                arg = np.array(arg).reshape(1,1)
            if arg.ndim == 1:
                arg = arg[:,np.newaxis]
            array_args.append( matlab.double(arg.tolist()) )
        result = getattr(self.eng, self.function_name)(*array_args, **kwargs)
        self.function_name = None
        return np.array( result )

    def __getattr__(self, attr):
        self.function_name = attr
        return self.__handler


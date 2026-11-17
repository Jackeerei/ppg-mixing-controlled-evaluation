"""Elapsed awake time, excluding Windows sleep; inherited past counters are never refunded."""
import os,time
ORIGINAL_MONOTONIC=time.monotonic
if os.name=='nt':
    import ctypes
    _library=ctypes.WinDLL('kernel32',use_last_error=True)
    try:_api=_library.QueryUnbiasedInterruptTimePrecise;KIND='QueryUnbiasedInterruptTimePrecise';_returns_bool=False
    except AttributeError:_api=_library.QueryUnbiasedInterruptTime;KIND='QueryUnbiasedInterruptTime';_returns_bool=True
    _api.argtypes=[ctypes.POINTER(ctypes.c_ulonglong)];_api.restype=ctypes.c_int if _returns_bool else None
    def seconds():
        value=ctypes.c_ulonglong();result=_api(ctypes.byref(value))
        if _returns_bool and not result:raise ctypes.WinError(ctypes.get_last_error())
        return value.value/10000000
else:seconds=ORIGINAL_MONOTONIC;KIND='monotonic'

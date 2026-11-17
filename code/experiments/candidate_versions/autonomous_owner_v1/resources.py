"""Small stdlib resource diagnostics; no CUDA/Torch context allocation."""
import ctypes
import datetime
import shutil
import subprocess
import psutil


class PerformanceInfo(ctypes.Structure):
    _fields_=[('cb',ctypes.c_uint32)]+[(n,ctypes.c_size_t) for n in
        ('CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable','SystemCache',
         'KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[(n,ctypes.c_uint32) for n in
        ('HandleCount','ProcessCount','ThreadCount')]


def snapshot(root):
    p=PerformanceInfo();p.cb=ctypes.sizeof(p)
    api=ctypes.WinDLL('psapi',use_last_error=True).GetPerformanceInfo
    api.argtypes=[ctypes.POINTER(PerformanceInfo),ctypes.c_uint32];api.restype=ctypes.c_int
    if not api(ctypes.byref(p),p.cb):raise ctypes.WinError(ctypes.get_last_error())
    try:
        q=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,timeout=5,
                         creationflags=subprocess.CREATE_NO_WINDOW)
        gpu_free=int(q.stdout.splitlines()[0]) if q.returncode==0 and q.stdout.strip() else None
        gpu_error=None
    except (subprocess.TimeoutExpired,OSError,ValueError,IndexError) as exc:
        q=None;gpu_free=None;gpu_error=repr(exc)
    value=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               available_ram_bytes=psutil.virtual_memory().available,
               commit_headroom_bytes=(p.CommitLimit-p.CommitTotal)*p.PageSize,
               commit_total_bytes=p.CommitTotal*p.PageSize,commit_limit_bytes=p.CommitLimit*p.PageSize,
               disk_free_bytes=shutil.disk_usage(root).free,
               gpu_free_mib=gpu_free,nvidia_smi_exit_code=q.returncode if q is not None else None,
               nvidia_smi_error=gpu_error)
    value['checks']={'ram_3_5_GiB':value['available_ram_bytes']>=3.5*1024**3,
                     'commit_headroom_2_GiB':value['commit_headroom_bytes']>=2*1024**3,
                     'disk_10_GiB':value['disk_free_bytes']>=10*1024**3,
                     'gpu_free_2048_MiB':value['gpu_free_mib'] is not None and value['gpu_free_mib']>=2048}
    return value


def admissible(observation):
    return all(observation['checks'].values())


def parallel_ready(observation):
    """Measured second-worker admission: peak +25% +4GiB reserve, rounded up."""
    return (admissible(observation) and observation['available_ram_bytes']>=6*1024**3
            and observation['commit_headroom_bytes']>=8*1024**3)

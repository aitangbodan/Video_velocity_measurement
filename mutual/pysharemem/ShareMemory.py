
import ctypes
import signal
import json

print("[Python]: 开始加载共享内存动态库...")
libLoad = ctypes.cdll.LoadLibrary
try:
    # module_root_path = os.path.dirname(__file__)
    # share = libLoad(os.path.join(module_root_path, "libsharememory.so"))
    from model.lib import sharelib as share
    print("[Python]: 加载共享内存动态库成功！")
    share.image_data_address.restype = ctypes.POINTER(ctypes.c_uint8)
    share.json_data_address.restype = ctypes.POINTER(ctypes.c_uint8)
    share.config_info_address.restype = ctypes.POINTER(ctypes.c_uint8)
except Exception as e:
    print("[Python]: 加载共享内存动态库失败！可能是由于没有将 libsharememory.so 放置在PyShareMemory模块目录下！")
    print("[Python]: 详细的错误信息-" + str(e))
    exit(1)


def blank_work():
    return "[来自Python的信息]: Null，没有传入工作函数！"


class ShareMemory(object):
    work = blank_work()

    def __init__(self, share_memory_key: int, slave_pid: int):
        share.create_share(share_memory_key, slave_pid)
        signal.signal(signal.SIGUSR1, self.job)
        pass

    def do(self, work_to_do):
        self.work = work_to_do

    def job(self, signum, frame):
        share.set_status_working()
        print('[Python]: 接收到传输完成的信号，开始处理...')
        res = self.work(self.get_data(), self.get_json())
        print('[Python]: 处理完成，开始向内存写入结果')
        share.write_result(ctypes.c_char_p(res.encode('utf-8')))
        share.set_status_done()
        print('[Python]: 写入结束，内存标志位已修改为JOB_DONE')
        print('[Python]: 本次任务结束。')

    # 从共享内存中获取数据
    def get_data(self):
        share_body_ptr = share.image_data_address()
        py_data_recv = ctypes.cast(share_body_ptr, ctypes.POINTER(ctypes.c_uint8 * share.image_data_size())).contents
        return py_data_recv

    # 获取json数据
    def get_json(self):
        share_json_ptr = share.json_data_address()
        py_json_recv = ctypes.cast(share_json_ptr, ctypes.POINTER(ctypes.c_uint8 * share.json_data_size())).contents
        # 转换为字符串
        print(bytearray(py_json_recv))
        json_str = bytearray(py_json_recv).decode('utf-8')
        
        print(json_str)
        # 转换为json对象
        json_ms = json.loads(json_str)
        return json_ms

    # 获取配置
    def get_config(self):
        share_config_ptr = share.config_info_address()
        py_config_recv = ctypes.cast(share_config_ptr, ctypes.POINTER(ctypes.c_uint8 * share.json_data_size())).contents
        # 转换为字符串
        config_str = bytearray(py_config_recv).decode('utf-8')
        # 转换为json对象
        if config_str == "":
            return json.loads("{}")
        config_ms = json.loads(config_str)
        return config_ms

    def host_pid(self):
        return share.host_pid()
    pass

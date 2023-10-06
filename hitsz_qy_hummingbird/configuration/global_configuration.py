import logging
import time
import os
import inspect


class MyLogger(logging.Logger):
    def debug(self,
              msg,
              *args,
              **kwargs):
        class_name = self._get_current_class_name()
        msg_with_class = f'[{class_name}] {msg}'
        super().debug(msg_with_class, *args, **kwargs)

    @staticmethod
    def _get_current_class_name():
        stack = inspect.stack()
        frame = stack[1].frame
        code = frame.f_back.f_code
        module = inspect.getmodule(code)
        classes = inspect.getmembers(module, inspect.isclass)

        for name, cls in classes:
            if code.co_name in cls.__dict__:
                return name

        return None


class GlobalConfigWrapper:
    def __init__(self):
        self.hash_code = 0
        self.logger = None

        # change the relative path to absolute path
        module = inspect.getmodule(GlobalConfigWrapper)
        module_file = module.__file__

        # make sure that the path is absolute
        absolute_path = os.path.abspath(module_file)
        absolute_dir_path = os.path.dirname(absolute_path)

        self.temporary_data_path = absolute_dir_path + "/../../TemporaryData/"
        self.temporary_urdf_path = absolute_dir_path + "/../../TemporaryURDF/"
        self.urdf_folder_path = absolute_dir_path + "/../../URDFdir/"
        self.logger_path = 0

        self.TIMESTEP = 5000
        self.TICKTOCK = 0
        self.TIME = 0

    def loginit_ex(self,
                   additional_file_folder,
                   readme):
        self.temporary_data_path = self.temporary_data_path + additional_file_folder + '/'
        self.logger_init()
        self.logger.debug(readme)

    def step(self):
        self.TICKTOCK = self.TICKTOCK + 1
        self.TIME = self.TICKTOCK / self.TIMESTEP

    def logger_init(self):
        """
        init the universal logger
        """

        current_time = int(time.time())
        local_time = time.localtime(current_time)

        self.hash_code = time.strftime("%Y%m%d_%H_%M_%S", local_time)
        self.logger = MyLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # 创建控制台 Handler，并设置日志级别为 DEBUG
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        self.logger_path = os.path.join(self.temporary_data_path, self.hash_code) + "/"
        os.makedirs(self.logger_path)

        # 创建文件 Handler，并设置日志级别为 DEBUG
        file_handler = logging.FileHandler(self.logger_path + 'logger.log')
        file_handler.setLevel(logging.DEBUG)

        # 定义日志输出格式
        formatter = logging.Formatter('%(asctime)s -  %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        # remark = input("Input The Running Info For logging：")
        #
        # # 将备注写入README文件
        # with open(self.logger_path + "README.txt", 'a') as f:
        #     f.write(remark + '\n')

        # 将 Handler 添加到 Logger 对象中
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

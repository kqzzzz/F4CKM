import logging
import time
from tqdm import tqdm
import io
import os
import sys

class TqdmLogger(io.StringIO):
    def __init__(self, log_file, bar_stdout_flag=True):
        super(TqdmLogger, self).__init__()
        self.logger = log_file
        self.buf = ''
        self.bar_index = -1
        self.bar_stdout_flag = bar_stdout_flag

        # Create the log file if it doesn't already exist
        if not os.path.isfile(self.logger):
            with open(self.logger, 'w', encoding='utf-8') as fstream:
                fstream.close()

    def reset(self):
        self.bar_index = -1
        self.buf = ''

    def write(self, buf, log_type=None):
        if buf.strip() and (log_type == 'text' or ('it/s' not in buf.strip() and 's/it' not in buf.strip())):
            with open(self.logger, 'a', encoding='utf-8') as ofile:
                ofile.write(buf)
                ofile.flush()
            sys.stdout.write(buf.rstrip() + '\n')
            sys.stdout.flush()
        else:
            self.buf = buf.strip('\r\n\t ')
            if self.bar_stdout_flag:
                sys.stdout.write(buf)
                sys.stdout.flush()

    def flush(self):
        if not self.buf.strip():
            return
        sys.stdout.flush()

        # Read the entire content of the log file
        with open(self.logger, 'r', encoding='utf-8') as ifstream:
            lines = ifstream.readlines()

        # Replace the progress bar line with new value
        if self.bar_index != -1:
            del lines[self.bar_index]
        lines.append(self.buf.strip() + '\n')
        self.bar_index = len(lines) - 1

        # Write back everything
        with open(self.logger, 'w', encoding='utf-8') as ofstream:
            ofstream.writelines(lines)

        # Reset the buffer for the next cycle
        self.buf = ''
        
# 配置日志
log_file = 'result.log'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_file, encoding='utf-8', mode='a')
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# 创建 TqdmLogger 实例
tqdm_stream = TqdmLogger(log_file)

# 重定向 tqdm 输出
tqdm_stream.reset()
pbar = tqdm(total=100, unit='iter', file=tqdm_stream)

# 示例循环
for i in range(100):
    # 模拟一些工作
    time.sleep(0.1)
    pbar.update(1)

pbar.close()
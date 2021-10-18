import yaml


class YamlHandler:
    def __init__(self, file):
        self.file=file
    def read_yaml(self, encoding='utf-8'):
        """读取yaml数据"""
        with open(self.file, encoding=encoding) as f:
            return yaml.safe_load(f.read())

    def write_yaml(self, data, encoding='utf-8'):
        """向yaml文件写入数据"""
        with open(self.file, encoding=encoding, mode='w') as f:
            return yaml.safe_dump(data, stream=f, sort_keys=False,default_flow_style=False)
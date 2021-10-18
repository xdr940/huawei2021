from utils.yaml_wrapper import YamlHandler




if __name__ == '__main__':

    # 读取config.yaml配置文件数据
    args = YamlHandler('./opts/mc_mbs.yaml').read_yaml()
    print(args)


    # 将data数据写入config1.yaml配置文件
    write_data = YamlHandler('./train_mc2.yaml').write_yaml(args)
    print(args)
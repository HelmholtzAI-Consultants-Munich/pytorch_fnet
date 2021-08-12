import os
from argparse import ArgumentParser
import yaml
import csv
import json

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help="training configuration")

# get config
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, yaml.SafeLoader)

def make_dataset_csv(config):
    file_dir = os.path.join(config['data_path'], config['dataset']) #config['path_dataset']
    write_file = os.path.join(config['data_path'], 'csvs', ('.').join([config['dataset'],'csv'])) #config['path_dataset_csv']
    channel_id, target_id, dapi_id, data_id = config['signal_channel'], config['target_channel'], config['dapi_channel'], config['dataset']

    with open(write_file, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['file','signal_channel', 'target_channel', 'dapi_channel', 'dataset'])

        for (dirpath, dirnames, filenames) in os.walk(file_dir):
            for file in filenames:
                if file.endswith('.tif'):
                    filename = os.path.join(dirpath, file)
                    writer.writerow([filename, channel_id, target_id, dapi_id, data_id])
    csv_file.close()

if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.config)
    
    path_dataset_csv = os.path.join(config['data_path'], 'csvs', ('.').join([config['dataset'],'csv'])) 
    if not os.path.isfile(path_dataset_csv):
        make_dataset_csv(config)
    
    new_csvs_path = os.path.join(config['data_path'], 'csvs', config['dataset'])
    command_str = f"python scripts/python/split_dataset.py {path_dataset_csv} {new_csvs_path} --train_size {config['train_size']} -v"
    os.system(command_str)
    
    config['path_run_dir'] = os.path.join(config['output_path'], config['dataset'], 'results_pretrained')
    temp_json_config = './temp.json'
    with open(temp_json_config, "w") as fp:
        json.dump(config, fp)
    command_str = f"python predict.py {temp_json_config}"
    os.system(command_str)
    os.remove(temp_json_config)



import os
from argparse import ArgumentParser
import yaml
import csv
import json


def get_config(config):
    '''
    Reads the config file and loads into a dict

    Parameters
    ----------
    config : str
        The config file - usually config.yaml
    Returns
    -------
    dict
        The loaded config file
    '''
    with open(config, 'r') as stream:
        return yaml.load(stream, yaml.SafeLoader)

def make_dataset_csv(config):
    '''
    This function creates a csv file with the path of the images in the dataset and the channel id of the brightfield, 
    DAPI and infection channels. It saves the csv file in a folder named 'csvs' within your config['data_path'] directory.
    
    Parameters
    ----------
    config : dict
        The config file - usually config.yaml
    '''
    
    file_dir = os.path.join(config['data_path'], config['dataset']) 
    if not os.path.exists(os.path.join(config['data_path'], 'csvs')):
        os.mkdir(os.path.join(config['data_path'], 'csvs'))
        
    write_file = os.path.join(config['data_path'], 'csvs', ('.').join([config['dataset'],'csv'])) 
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

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help="training configuration")

    args = parser.parse_args()
    config = get_config(args.config) # get the config.yaml file as a dictionary
    
    # create a csv file with list of images in dataset if it doesn't already exist
    path_dataset_csv = os.path.join(config['data_path'], 'csvs', ('.').join([config['dataset'],'csv'])) 
    if not os.path.isfile(path_dataset_csv):
        make_dataset_csv(config)
        
    # split that file into a train and test set - split % is defined by config['train_size']
    new_csvs_path = os.path.join(config['data_path'], 'csvs', config['dataset'])
    command_str = f"python scripts/python/split_dataset.py {path_dataset_csv} {new_csvs_path} --train_size {config['train_size']} -v"
    os.system(command_str)
    
    config['path_run_dir'] = [os.path.join(config['output_path'], config['dataset'], 'pretrained')]
    # save the config file to a temp.json file - used as argument to script call below
    temp_json_config = './temp.json'
    with open(temp_json_config, "w") as fp:
        json.dump(config, fp)
        
    # and run pre-trained model on test set
    command_str = f"python predict.py {temp_json_config}"
    os.system(command_str)
    os.remove(temp_json_config)
import os
from argparse import ArgumentParser
import yaml
import json
import numpy as np
from fnet.functions import compute_dataset_min_max_ranges

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

def get_best_model(run_paths_list):
    '''
    Returns the best model from a list of models

    This function computes the average pearson correlation coefficient for each model
    and returns the best performing model and pearson value for that model

    Parameters
    ----------
    run_paths_list : list
        A list of paths of different model runs
    Returns
    -------
    int
        The id of the best model
    float
        The pearson correlation coefficient of the best model
    '''
    # get average pearson for each model
    pearson_scores = np.zeros(len(run_paths_list))
    for idx, path_run_dir in enumerate(run_paths_list):
        with open(os.path.join(path_run_dir, 'results', 'pearson.json'), "r") as fp:
            pearson_model = json.load(fp)
            for param in pearson_model.values():
                pearson_scores[idx] += param  
    id_max = np.argmax(pearson_scores)
    return id_max, pearson_scores[id_max]/len(pearson_model)
   
if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help="training configuration")
    
    args = parser.parse_args()
    config = get_config(args.config) # get the config.yaml file as a dictionary

    # create a csv file with list of images in dataset if it doesn't already exist
    path_dataset_csv = os.path.join(config['data_path'], 'csvs', ('.').join([config['dataset'],'csv'])) 
    if not os.path.isfile(path_dataset_csv):
        make_dataset_csv(config)
    
    # define the paths where the data will be saved 
    path_dataset_csv_paths = os.path.join(config['data_path'], 'csvs', config['dataset']) 
    path_dataset_train_csv = os.path.join(path_dataset_csv_paths, 'train.csv') 
    # if train.csv doesn't exist already (from workflow_pt1) create it
    if not os.path.isfile(path_dataset_train_csv):
        command_str = f"python scripts/python/split_dataset.py {path_dataset_csv} {path_dataset_csv_paths} --train_size {config['train_size']} -v"
        os.system(command_str)
    
    # if dataset min-max has not been computed in workflow_pt1:
    if 'intensities' not in config:
        config['intensities'] = {}
        # get min max values this may be needed later for normalisation
        min_max_bright, min_max_infection, min_max_dapi = compute_dataset_min_max_ranges(path_dataset_train_csv)
        min_max_bright_norm, min_max_infection_norm, _ = compute_dataset_min_max_ranges(path_dataset_train_csv, norm=True)
        config['intensities']['max_infection'] = int(min_max_infection[1])
        config['intensities']['min_infection'] = int(min_max_infection[0])
        config['intensities']['max_norm_infection'] = float(min_max_infection_norm[1])
        config['intensities']['min_norm_infection'] = float(min_max_infection_norm[0])
        config['intensities']['max_brightfield'] = int(min_max_bright[1])
        config['intensities']['min_brightfield'] = int(min_max_bright[0])
        config['intensities']['max_norm_brightfield'] = float(min_max_bright_norm[1])
        config['intensities']['min_norm_brightfield'] = float(min_max_bright_norm[0])
        config['intensities']['max_dapi'] = int(min_max_dapi[1])
        config['intensities']['min_dapi'] = int(min_max_dapi[0])
        # update the config
        with open(args.config, "w") as fp:
            json.dump(config, fp, indent=4)

    # create splits of dataset used for training into train-val - number of splits is defined by config['kfolds']
    command_str = f"python scripts/python/split_dataset.py {path_dataset_train_csv} {path_dataset_csv_paths} --kfolds {config['kfolds']} --val_split {config['val_size']} -v"
    os.system(command_str)
    
    # save the config file to a temp.json file - used as argument to script call below
    temp_json_config = './temp.json'
    with open(temp_json_config, "w") as fp:
        json.dump(config, fp)
    
    # models holds paths of all models which will be compared in predict later
    #models = [config['path_model_dir'][0]]
    pretrained_results_path = os.path.join(config['output_path'], config['dataset'], 'pretrained')
    config['path_run_dir'] = [pretrained_results_path]

    if config['training']['method'] == 'fine-tune' or config['training']['method'] == 'both':
        # run a training experiment while finetuning the existing model
        command_str = f"python train_experiment.py {temp_json_config} --fine_tune --verbose"
        os.system(command_str)
        output_path_finetune = os.path.join(config['output_path'], config['dataset'], config['run'], 'fine_tuned')
        config['path_model_dir'].append(output_path_finetune) # models
        config['path_run_dir'].append(output_path_finetune)

    if config['training']['method'] == 'from-scratch' or config['training']['method'] == 'both':
        # run a training experiment by training the model from scratch
        command_str = f"python train_experiment.py {temp_json_config} --verbose"
        os.system(command_str)
        output_path_train_scratch = os.path.join(config['output_path'], config['dataset'], config['run'], 'train_from_scratch')
        config['path_model_dir'].append(output_path_train_scratch) # models
        config['path_run_dir'].append(output_path_train_scratch)
   
    # store pearson correlation metrics for each model on test set
    config['prediction']['return_score'] = True 
    #config['path_model_dir'] = models
    
    # update the stored config
    temp_json_config = './temp.json'
    with open(temp_json_config, "w") as fp:
        json.dump(config, fp)
        
    # and run all models on test set while storing pearson correlation coefficient results for each model
    command_str = f"python predict.py {temp_json_config}"
    os.system(command_str)
    
    # find the model with the best pearson correlation coefficient
    id_max, pearson_score = get_best_model(config['path_run_dir'])
    print("THE BEST MODEL IS: {} WITH A PEARSON CORRELATION COEFFICIENT OF: {}".format(config['path_model_dir'][id_max], pearson_score))
    print("RESULTS FROM THIS MODEL CAN BE FOUND IN: ", ('/').join([config['path_model_dir'][id_max], 'results']))
    
    os.remove(temp_json_config) # remove the temp file
import os
from argparse import ArgumentParser
import yaml
import json
import numpy as np

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
    
    # define the paths where the data will be saved 
    path_dataset_csv_paths = os.path.join(config['data_path'], 'csvs', config['dataset']) 
    path_dataset_train_csv = os.path.join(path_dataset_csv_paths, 'train.csv') 
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
import os
from argparse import ArgumentParser
import yaml
import json

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help="training configuration")

# get config
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, yaml.SafeLoader)

if __name__ == '__main__':
    args = parser.parse_args()
    config = get_config(args.config)
    
    path_dataset_csv_paths = os.path.join(config['data_path'], 'csvs', config['dataset']) #./data/csvs/new_exp_eg
    path_dataset_train_csv = os.path.join(path_dataset_csv_paths, 'train.csv') #"./data/csvs/new_exp_eg/train.csv"
    command_str = f"python scripts/python/split_dataset.py {path_dataset_train_csv} {path_dataset_csv_paths} --kfolds {config['kfolds']} -v"
    os.system(command_str)

    temp_json_config = './temp.json'
    with open(temp_json_config, "w") as fp:
        json.dump(config, fp)

    command_str = f"python train_experiment.py {temp_json_config} --fine_tune --verbose"
    os.system(command_str)
    
    command_str = f"python train_experiment.py {temp_json_config} --verbose"
    os.system(command_str)

    output_path_finetune = os.path.join(config['output_path'], config['dataset'], config['run'], 'fine_tuned')
    output_path_train_scratch = os.path.join(config['output_path'], config['dataset'], config['run'], 'train_from_scratch')

    config['prediction']['return_score'] = True
    models = [output_path_finetune, output_path_train_scratch, config['path_model_dir']]
    config['path_model_dir'] = models
    pretrained_results_path = os.path.join(config['output_path'], config['dataset'], 'pretrained')
    config['path_run_dir'] = [output_path_finetune, output_path_train_scratch, pretrained_results_path ]
    command_str = f"python predict.py {temp_json_config}"
    pearson_scores = os.system(command_str)
    id_max = pearson_scores.argmax()
    print("THE BEST MODEL IS: ", models[id_max])
    print("RESULTS FROM THIS MODEL CAN BE FOUND IN: ", ('/').join([models[id_max], 'results']))
    
    os.remove(temp_json_config)
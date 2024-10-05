import json 

def load_json(path):
    '''
    Simple function that reads json file into a python dictionary.
    '''
    with open(path, "r") as file:
        map = json.load(file)
    return map


def swap_dataset(new_dataset_path):
    '''
    Simple function that changes the dataset path in the paths.json config file. Used to switch between using the ChemProt_Reduced.csv data versus the ChemProt.csv data. 
    '''
    config = load_json("config_files/paths.json")
    config["dataset"] = new_dataset_path
    with open("config_files/paths.json", "w") as file:
        json.dump(config, file)
    return

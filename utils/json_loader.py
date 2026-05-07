import os
import json
from pathlib import Path


def load_config(filepath: Path | str) -> dict:
    try:
        with open(file=filepath, mode='r', encoding='utf-8') as file:
            data = json.load(file)
        
    except json.JSONDecodeError as e:
        print('[!] config.json format is wrong. Fix or delete it to create a new one.')
        raise 
    
    except FileNotFoundError:
        #raise FileNotFoundError('config.json not found. Please edit the file.')
        #_create_config(filepath)
        absolute_path = os.path.abspath(filepath)
        print(f'[!] config.json not found. Please edit the file "{absolute_path}".')
        raise 
    
    return data
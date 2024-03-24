import torch
import os
import re

if __name__ == '__main__':
    model_path = './lora_0/lora_0_2'
    pattern = re.compile(r'^adapter_model_.*\.bin$')
    files = os.listdir(model_path)
    matched_files = [file for file in files if pattern.match(file)]

    total_model = {}
    for file in matched_files:
        path = os.path.join(model_path, file)
        model = torch.load(path, map_location='cpu')
        total_model.update(model)

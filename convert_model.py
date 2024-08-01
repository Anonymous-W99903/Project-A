"""
In torch where the version is higher than 1.6, the model is saved as a zip archive.
This would lead to an error when the zip model is loaded by torch below 1.6.
The script is for converting the zip model into no-zip model.
"""
import torch

def convert(PATH, NEW_PATH):
    state_dict = torch.load(PATH, map_location="cuda:0")
    torch.save(state_dict, NEW_PATH, _use_new_zipfile_serialization=False)
    
# convert(f'baseline3/checkpoint_current.pth.tar', 
#             f'baseline3/checkpoint_current.pth.tar') # overwrite
for i in [5] + list(range(10,17)):
    for j in [1,2,3]:
        convert(f'./results/widar3_new/person/baseline{i}_r{j}/checkpoint_current.pth.tar', 
                f'./results/widar3_new/person/baseline{i}_r{j}/checkpoint_current.pth.tar') # overwrite
        print(f'sucessfully convert {i},{j}')
    
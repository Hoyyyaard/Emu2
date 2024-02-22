import os
import json
from tqdm import tqdm
from torch.utils.data import Dataset
import random
from PIL import Image
import sys
import glob
# Abs file dir of this file
current_file_path = os.path.abspath(__file__)
# parent directory of this file
parent_directory = os.path.dirname(current_file_path)
base_dir = os.path.dirname(parent_directory)
# print(base_dir)
sys.path.append(base_dir)
import random
import numpy as np
import torch

class Diffusion_Finetune_Dataset(Dataset):
    '''
        This dataset will leverage different preprocessed dataset:
            1. EgoExo4d_Finetune_Dataset
    '''
    def __init__(self, split='train', preprocess_func=None, dataset_list= ['egoexo']):
        self.EgoExo4d_Pretrain_dataset_path = os.path.join('datasets', 'EgoExo4d', 'preprocessed_episodes', split)
        self.Epic_Kitchen_Text_Image_Pairs_dataset_path = os.path.join('datasets', 'epic-kitchen', 'text_image_pairs', split)
        self.preprocess_func = preprocess_func
        self.episodes = []
        os.makedirs(os.path.join('datasets', 'EgoExo4d', 'save_data'), exist_ok=True)
        if 'egoexo' in dataset_list:
            saved_data_path = os.path.join('datasets', 'EgoExo4d', 'save_data', f'image_text_pairs_{split}_emu2.pkl')
            if os.path.exists(saved_data_path):
                print("Loading saved data...")
                self.recover_data(saved_data_path)
                print("Loaded saved data for EgoExo4d_Pretrain_Dataset!")
            else:
                print("Loading EgoExo4d_Pretrain_Dataset..")
                self.load_from_EgoExo4d_Pretrain_Dataset()
                self.save_process_data(saved_data_path)
    
    def recover_data(self, saved_file):
        all_data = torch.load(saved_file)
        self.episodes = all_data['episodes']
        del all_data
        
    def save_process_data(self, saved_file):
        all_data = {'episodes': self.episodes}
        torch.save(all_data, saved_file)
       
    def load_from_EgoExo4d_Pretrain_Dataset(self, ):
        # Each episode will be in format {'image_path', 'caption'}
        for task_name in tqdm(os.listdir(self.EgoExo4d_Pretrain_dataset_path), desc='Loading EgoExo4d_Pretrain_Dataset'):
            take_path = os.path.join(self.EgoExo4d_Pretrain_dataset_path, task_name)
            for frame_path in os.listdir(take_path):
                with open(os.path.join(take_path, frame_path, 'caption.json'), 'r') as f:
                    data = json.load(f)
                    caption = data['caption']
                image_path = os.path.join(take_path, frame_path, 'ego_rgb.png')
                exo_path = glob.glob(os.path.join(take_path, frame_path, 'cam*.png'))
                self.episodes.append({'image_path':image_path, 'caption':caption, 'exo_path':exo_path})
                
    def load_from_Epic_Kitchen_Text_Image_Pairs_Dataset(self, ):
        for video_index in tqdm(os.listdir(self.Epic_Kitchen_Text_Image_Pairs_dataset_path), desc='Loading Epic_Kitchen_Text_Image_Pairs_Dataset'):
            for index in (os.listdir(os.path.join(self.Epic_Kitchen_Text_Image_Pairs_dataset_path, video_index))):
                path = os.path.join(self.Epic_Kitchen_Text_Image_Pairs_dataset_path, video_index, index)
                with open(os.path.join(path, 'caption.txt'), 'r') as f:
                    caption = f.read()
                image_path = os.path.join(path, 'image.png')
                self.episodes.append({'image_path':image_path, 'caption':caption})
        
    
    def __getitem__(self, i):
        image_p, text = self.episodes[i]['image_path'], self.episodes[i]['caption']
        image = Image.open(image_p).convert("RGB")
        pixel_values, input_ids = self.preprocess_func(image, text)
        
        exo_path = self.episodes[i]['exo_path']
        exo_images = []
        for p in exo_path:
            exo_images.append(Image.open(p).convert("RGB"))
        exo_pixel_values = []
        for exo in exo_images:
            exo_pixel_values.append(self.preprocess_func(exo, text)[0])
        exo_pixel_values = torch.stack(exo_pixel_values)
        return {'pixel_values':pixel_values, 
                'input_ids':input_ids,
                'image':image,
                'original_image': exo_images,
                'text':text,
                'exo_pixel_values':exo_pixel_values,
                'original_pixel_values': exo_pixel_values,  
                'interleave_sequence': exo_images + [text] + [image]}
                
        
    def __len__(self):
        return len(self.episodes)   
        


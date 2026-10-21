import warnings
import sys
sys.path.append("./saliency_maps/model")  # 添加本地模型路径

from modeling_biomed_clip import BiomedCLIPModel, BiomedCLIPConfig  # 本地导入
warnings.filterwarnings('ignore')
import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from scripts.clip_wrapper import ClipWrapper
from scripts.plot import visualize_vandt_heatmap
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import itertools
import torch
import json
import random
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizerFast,CLIPFeatureExtractor
from PIL import Image
from scripts.methods import vision_heatmap_iba
from text_prompts import *
# Disable parallel tokenization warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Function to calculate Dice coefficient for evaluating segmentation
def calculate_dice_coefficient(mask1, mask2):
    # Calculate intersection and Dice score
    intersection = np.logical_and(mask1, mask2).sum()
    dice_coefficient = (2.0 * intersection) / (mask1.sum() + mask2.sum())
    return dice_coefficient

# Function to evaluate a model on a sample image and calculate the Dice score
def evaluate_on_sample(model, processor, tokenizer, text, image_paths, args):
    dice_scores = []
    for image_id in tqdm(image_paths):
        try:
            image = Image.open(f"{args.val_path}/{image_id}").convert('RGB')
        except:
            continue

        # 分别处理图像和文本
        image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(args.device)
        text_ids = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")['input_ids'].to(args.device)

        vmap = vision_heatmap_iba(
            text_ids, image_feat, model,
            args.vlayer, args.vbeta, args.vvar,
            ensemble=args.ensemble, progbar=False
        )

        gt_path = args.val_path.replace("images", "masks")
        gt_mask = np.array(Image.open(f"{gt_path}/{image_id}").convert("L"))

        vmap_resized = cv2.resize(np.array(vmap), (gt_mask.shape[1], gt_mask.shape[0]))
        cam_img = vmap_resized > 0.3

        dice_score = calculate_dice_coefficient(gt_mask.astype(bool), cam_img.astype(bool))
        dice_scores.append(dice_score)

    return np.mean(dice_scores)

# Function to perform hyperparameter optimization
def hyper_opt(model, processor, tokenizer, text, args):
    print("Running Hyperparameter Optimization ...")

    # Define lists of possible hyperparameter values
    vbeta_list = [0.1, 1.0, 2.0]
    vvar_list = [0.1, 1.0, 2.0]
    layers_list = [7, 8, 9]

    # Create all combinations of the hyperparameters
    hyperparameter_combinations = list(itertools.product(vbeta_list, vvar_list, layers_list))

    # Get all image IDs from the validation path
    all_image_ids = sorted(os.listdir(args.val_path))

    results = []  # Store results of each combination

    # Iterate through each hyperparameter combination
    for combo in hyperparameter_combinations:
        vbeta, vvar, layer = combo
        args.vbeta = vbeta
        args.vvar = vvar
        args.vlayer = layer
        
        sample_dice_scores = []  # Store Dice scores for each sample

        print(f"Evaluating combination: vbeta={vbeta}, vvar={vvar}, layer={layer}")

        # Run 3 random samples to get an average performance
        for i in range(3):
            random.seed(i)
            sampled_images = random.sample(all_image_ids, 1)
            avg_dice = evaluate_on_sample(model, processor, tokenizer, text, sampled_images, args)
            sample_dice_scores.append(avg_dice)
            print(f"  Sample {i+1}: Average Dice Score = {avg_dice}")

        # Calculate mean Dice score for this hyperparameter combination
        mean_dice = np.mean(sample_dice_scores)
        results.append({
            'vbeta': vbeta,
            'vvar': vvar,
            'vlayer': layer,
            'average_dice': mean_dice
        })
        print(f"Mean Dice Score for this combination: {mean_dice}\n")

    # Convert results to a DataFrame for easy analysis
    results_df = pd.DataFrame(results)

    # Find the combination with the best Dice score
    best_combo = results_df.loc[results_df['average_dice'].idxmax()]
    
    print("Best Hyperparameter Combination:")
    print(best_combo)
    print("\n")
    
    return best_combo

# Main function to load the model, handle input/output, and generate saliency maps
def main(args):

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print("Loading models ...")
    
    # Load the appropriate model based on the arguments
    if(args.model_name == "BiomedCLIP" and args.finetuned):
        # model = AutoModel.from_pretrained("./saliency_maps/model", trust_remote_code=True).to(args.device)
        model = BiomedCLIPModel.from_pretrained("./saliency_maps/model").to(args.device)
        # model_path = "/home/rxk/.cache/huggingface/hub/models--chuhac--BiomedCLIP-vit-bert-hf/snapshots/c44fed22e30460c0ea1b033a41b68fdab375d64c"
        # processor = AutoProcessor.from_pretrained(model_path)
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            "/home/rxk/.cache/huggingface/hub/models--chuhac--BiomedCLIP-vit-bert-hf/snapshots/c44fed22e30460c0ea1b033a41b68fdab375d64c",
            local_files_only=True
        )
        processor = CLIPFeatureExtractor.from_pretrained(
            "/home/rxk/.cache/huggingface/hub/models--chuhac--BiomedCLIP-vit-bert-hf/snapshots/c44fed22e30460c0ea1b033a41b68fdab375d64c",
            local_files_only=True
        )
        # processor = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf",  local_files_only=False)
        # tokenizer = AutoTokenizer.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf",  local_files_only=False)
    elif(args.model_name == "BiomedCLIP" and not args.finetuned):
        model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True).to(args.device)
        processor = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    elif(args.model_name == "CLIP" and not args.finetuned):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(args.device)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch16")
    elif(args.model_name == "CLIP" and args.finetuned):
        model = AutoModel.from_pretrained("./model", trust_remote_code=True).to(args.device)

    if(not args.reproduce):
        # Get user input for the text to generate saliency maps
        text = str(input("Enter the text: "))

    # Perform hyperparameter optimization if required
    if(args.hyper_opt):
        best_combo = hyper_opt(model, processor, tokenizer, text, args)
        args.vbeta = best_combo['vbeta']
        args.vvar = best_combo['vvar']
        args.vlayer = int(best_combo['vlayer'])
        
    print("Generating Saliency Maps ...")

    # Create the output directory if it does not exist
    if(not os.path.exists(args.output_path)):
        os.makedirs(args.output_path)

    # Iterate through the input images and generate saliency maps
    for image_id in tqdm(sorted(os.listdir(args.input_path))):
        if (image_id in os.listdir(args.output_path)):
            continue
        try:
            image = Image.open(f"{args.input_path}/{image_id}").convert('RGB')
        except:
            print(f"Unable to load image at {image_id}", flush=True)
            continue

        if (args.reproduce):
            with open(args.json_path) as json_file:
                json_decoded = json.load(json_file)
            text = json_decoded[image_id]

        # 分开处理文本和图像
        image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(args.device)
        text_ids = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt")['input_ids'].to(
            args.device)

        vmap = vision_heatmap_iba(text_ids, image_feat, model, args.vlayer, args.vbeta, args.vvar,
                                  ensemble=args.ensemble, progbar=False)

        img = np.array(image)
        vmap = cv2.resize(np.array(vmap), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"{args.output_path}/{image_id}", vmap * 255)

# Entry point for the script
if __name__ == '__main__':
    # Define argument parser for input/output paths and hyperparameters
    parser = argparse.ArgumentParser('M2IB argument parser')
    parser.add_argument('--input-path', required=True, default="data/input_images", type=str, help='path to the images')
    parser.add_argument('--output-path', required=True, default="saliency_map_outputs", type=str, help='path to the output')
    parser.add_argument('--val-path', type=str, default="data/val_images", help='path to the validation set for hyperparameter optimization')
    parser.add_argument('--vbeta', type=float, default=0.1)
    parser.add_argument('--vvar', type=float, default=1.0)
    parser.add_argument('--vlayer', type=int, default=7)
    parser.add_argument('--tbeta', type=float, default=0.3)
    parser.add_argument('--tvar', type=float, default=1)
    parser.add_argument('--tlayer', type=int, default=9)
    parser.add_argument('--model-name', type=str, default="BiomedCLIP", help="Which CLIP model to use")
    parser.add_argument('--finetuned', action='store_true', help="Whether to use finetuned weights or not")
    parser.add_argument('--hyper-opt', action='store_true', help="Whether to optimize hyperparameters or not")
    parser.add_argument('--device', type=str, default="cuda", help="Device to run the model on")
    parser.add_argument('--ensemble', action='store_true', help="Whether to use text ensemble or not")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument('--json-path', type=str, default="busi.json", help="Path to the JSON file containing the text prompts")
    parser.add_argument('--reproduce', action='store_true')
    args = parser.parse_args()
    main(args)

    print("Saliency Map Generation Done!")
    
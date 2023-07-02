from tqdm import tqdm
import gc
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from model import TripletLoss, TripletNet, Identity, create_embedding_net
from dataset import QueryExtractor, EmbeddingDataset
from torchvision import transforms
import torchvision.models as models
import torch
from utils import draw_label, ap_at_k_per_query, get_preds, get_gt_web, get_preds_and_visualize, perform_pca_on_single_vector, ap_per_query
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from inference import get_query_embedding


def getModel(weights_file="./static/weights/oxbuild_final.pth"):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    resnet_model = create_embedding_net()
    model = TripletNet(resnet_model)
    model.load_state_dict(torch.load(weights_file,map_location='cpu'))
    model.to(device)
    model.eval()
    return model

#Были удалены функция для валидации
################################################################ NAM NADO OSTSUDA####################################################
def inference_on_single_labelled_image_pca_web_original(model, query_img_file, directory,
                top_k=60,
                plot=False,
                ):
    
   
    fts_dir=directory+'features'
    image_dir=directory+'images'
    # Create cuda parameters
    use_cuda = torch.cuda.is_available()
    np.random.seed(2019)
    torch.manual_seed(2019)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Available device = ", device)
    # Creat image database
    QUERY_IMAGES_FTS=[os.path.join(fts_dir, file) for file in sorted(os.listdir(fts_dir))]
    QUERY_IMAGES= [os.path.join(image_dir, file) for file in sorted(os.listdir(image_dir))]
    query_fts =  get_query_embedding(model, device, "."+query_img_file).detach().cpu().numpy()
    query_fts = perform_pca_on_single_vector(query_fts)

    # Create similarity list
    similarity = []
    for file in tqdm(QUERY_IMAGES_FTS):
        file_fts = np.squeeze(np.load(file))
        cos_sim = np.dot(query_fts, file_fts)/(np.linalg.norm(query_fts)*np.linalg.norm(file_fts))
        similarity.append(cos_sim)

    # Get best matches using similarity
    similarity = np.asarray(similarity)
    indexes = (-similarity).argsort()[:top_k]
    best_matches = [QUERY_IMAGES[index] for index in indexes]
    #print(best_matches)
    for i in range(len(best_matches)):
        best_matches[i]=best_matches[i][1:]
    return best_matches

if __name__ == '__main__':
    pass
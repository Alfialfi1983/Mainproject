import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import glob
import torchvision
from.net import dehaze_net  # Assuming dehaze_net is your CNN model

# Load the pre-trained CNN model
model = dehaze_net()
model.load_state_dict(torch.load('C:/Users/afsal/PycharmProjects/imagedehyzeprg/imagedehyzeprg/hyzeapp/snapshots/dehazer.pth', map_location=torch.device('cpu')))

model.eval()

import os


def dehaze_image(image_path):
    # Load the image
    data_hazy = Image.open(image_path)
    data_hazy = (np.asarray(data_hazy) / 255.0)
    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.unsqueeze(0)

    with torch.no_grad():
        clean_image = model(data_hazy)

    # Convert the clean image to numpy array
    clean_image_np = clean_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    clean_image_pil = Image.fromarray((clean_image_np * 255).astype(np.uint8))

    # Save the dehazed image
    final_image_dir = os.path.join('media', 'final')
    os.makedirs(final_image_dir, exist_ok=True)  # Create the directory if it doesn't exist
    final_image_path = os.path.join(final_image_dir, 'final_image.png')
    clean_image_pil.save(final_image_path)

    return final_image_path
    clean_image_pil.save(final_image_path)


    return render(request, 'dehaze.html', context)
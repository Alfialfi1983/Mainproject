from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import cv2
import numpy as np
import io
from PIL import Image
import base64
from django.shortcuts import render
from subprocess import Popen
from .net import dehaze_net
import torch
from torchvision import transforms
model = dehaze_net()
model.load_state_dict(torch.load(r'C:\Users\afsal\PycharmProjects\imagedehyzeprg\imagedehyzeprg\hyzeapp\snapshots\dehazer.pth', map_location=torch.device('cpu')))
model.eval()
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate
from .models import CustomUser
from django.shortcuts import render
from django.http import JsonResponse
from .video_dehaze import dehaze_video
import os
@csrf_exempt
def dehaze_image(request):
    if request.method == 'POST':
        # Get the image file from the request
        image_file = request.FILES.get('image')
        if not image_file:
            return JsonResponse({'error': 'No image provided'})

        try:
            # Read the image file
            image_data = io.BytesIO(image_file.read())
            image = Image.open(image_data)

            # Convert the image to numpy array
            image_np = np.array(image)

            # Dehaze the image with DCP
            dcp_image = dehaze(image_np, model='dark_channel')
            dcp_image_pil = Image.fromarray(dcp_image)
            # Dehaze the DCP image with ASM
            asm_image = dehaze(dcp_image, model='atmospheric_scattering')

            # Dehaze the ASM image with multi-scale Retinex
            msr_image = dehaze(asm_image, model='multi_scale_retinex')

            # Perform image fusion (Gaussian-weighted image fusion) between DCP and multi-scale Retinex images
            fused_image = gaussian_weighted_image_fusion(dcp_image, msr_image)
            fused_image_pil = Image.fromarray(fused_image)

            # Apply Unsharp Masking for further detail enhancement and sharpening
            usm_image = unsharp_masking(fused_image)

            # Convert the Unsharp Masking result to PIL Image
            usm_image_pil = Image.fromarray(usm_image)

            # Save the Unsharp Masking result to BytesIO object in PNG format
            output_data = io.BytesIO()
            usm_image_pil.save(output_data, format='PNG')
            output_data.seek(0)
            base64_usm_image = base64.b64encode(output_data.getvalue()).decode('utf-8')

            # Save the final image on the device
            final_image_path = r'C:\Users\afsal\PycharmProjects\imagedehyzeprg\imagedehyzeprg\hyzeapp\media\final_image.png'
            fused_image_pil.save(final_image_path)

            # Call dehaze.py to perform CNN on the finalized image
            processed_image_path = process_final_image(image_data)

            # Convert the original image to base64
            image_data.seek(0)
            base64_hazy_image = base64.b64encode(image_data.getvalue()).decode('utf-8')

            # Convert the processed image to base64
            processed_image_data = io.BytesIO()
            processed_image_pil = Image.open(processed_image_path)
            processed_image_pil.save(processed_image_data, format='PNG')
            processed_image_data.seek(0)
            base64_processed_image = base64.b64encode(processed_image_data.getvalue()).decode('utf-8')

            # Return the hazy, dehazed, and processed images as base64 encoded strings
            return JsonResponse({
                'hazy_image': base64_hazy_image,
                'final_image': base64_usm_image,
                'processed_image': base64_processed_image
            })
        except Exception as e:
            return JsonResponse({'error': str(e)})
        else:
            return JsonResponse({'error': 'Method not allowed'}, status=405)

def process_final_image(dcp_image):
    # Load the final image for further processing
    final_image = Image.open(dcp_image)

    # Convert the final image to the format expected by the model
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Assuming your model expects 224x224 images
        transforms.ToTensor(),
        # Add any other transformations required by your model
    ])
    final_image_tensor = transform(final_image).unsqueeze(0)

    # Process the final image using your CNN model
    with torch.no_grad():
        output = model(final_image_tensor)

    # Convert the output tensor to an image
    processed_image_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
    processed_image_pil = Image.fromarray((processed_image_np * 255).astype(np.uint8))

    # Save the processed image
    processed_image_path = r'C:\Users\afsal\PycharmProjects\imagedehyzeprg\imagedehyzeprg\hyzeapp\media\result.png'
    processed_image_pil.save(processed_image_path)

    return processed_image_path

def unsharp_masking(image):
    # Apply Unsharp Masking
    blurred = cv2.GaussianBlur(image, (0, 0), 2)
    usm_image = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

    # Clip and convert the usm image to uint8
    usm_image = np.clip(usm_image, 0, 255).astype(np.uint8)

    return usm_image

def gaussian_weighted_image_fusion(image1, image2):
    alpha = 0.5  # Weighting factor for image1
    beta = 1 - alpha  # Weighting factor for image2
    fused_image = cv2.addWeighted(image1, alpha, image2, beta, 0)
    return fused_image


def dehaze(image, model='', omega=0.75, t0=0.1):
    if model == 'dark_channel':
        return dehaze_dark_channel(image, omega, t0)
    if model == 'atmospheric_scattering':
        return dehaze_atmospheric_scattering(image, omega, t0)
    if model == 'multi_scale_retinex':
        return dehaze_multi_scale_retinex(image)
    else:
        raise ValueError('Invalid dehazing model specified')

def dehaze_dark_channel(image, omega=0.85, t0=0.1):
    # Calculate the dark channel of the image
    dark_channel = np.min(image, axis=2)

    # Estimate the atmospheric light
    top_percentile = int(image.size * 0.001)
    dark_channel_flat = dark_channel.flatten()
    indices = np.argpartition(dark_channel_flat, -top_percentile)[-top_percentile:]
    atmospheric_light = np.max(image.reshape(-1, 3)[indices], axis=0)

    # Estimate the transmission map
    transmission_map = 1 - omega * dark_channel / atmospheric_light.max()

    # Clamp transmission values
    transmission_map = np.clip(transmission_map, t0, 1)

    # Compute the dehazed image
    dehazed_image = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        dehazed_image[:, :, i] = (image[:, :, i].astype(np.float32) - atmospheric_light[i]) / transmission_map + \
                                 atmospheric_light[i]

    # Further enhance the dehazed image (optional)
    # Apply gamma correction
    dehazed_image = np.power(dehazed_image / 255.0, 1.2) * 255.0
    dehazed_image = np.clip(dehazed_image, 0, 255).astype(np.uint8)

    return dehazed_image

def dehaze_atmospheric_scattering(image, omega=0.85, t0=0.1, beta=1.0):
    # Estimate the atmospheric light
    top_percentile = int(image.size * 0.001)
    dark_channel_flat = np.min(image, axis=2).flatten()
    indices = np.argpartition(dark_channel_flat, -top_percentile)[-top_percentile:]
    atmospheric_light = np.max(image.reshape(-1, 3)[indices], axis=0)

    # Calculate the distance from the camera to the scene
    distance = -beta * np.log(1 - omega * np.min(image / atmospheric_light, axis=2))

    # Estimate the transmission map
    transmission_map = 1 - omega * distance

    # Clamp transmission values
    transmission_map = np.clip(transmission_map, t0, 1)

    # Compute the dehazed image
    dehazed_image = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        dehazed_image[:, :, i] = (image[:, :, i].astype(np.float32) - atmospheric_light[i]) / transmission_map + \
                                 atmospheric_light[i]

    dehazed_image = np.clip(dehazed_image, 0, 255).astype(np.uint8)

    return dehazed_image


def dehaze_multi_scale_retinex(image, sigma_list=[15, 80, 250], gain=128, offset=0):
    # Multi-scale Retinex algorithm
    def single_scale_retinex(img, sigma):
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        # Calculate the retinex
        retinex = np.log10(img) - np.log10(blur)
        return retinex

    # Apply single-scale Retinex for each sigma value
    retinex_list = [single_scale_retinex(image.astype(np.float32) + offset, sigma) for sigma in sigma_list]

    # Combine the multi-scale Retinex results
    retinex = np.sum(retinex_list, axis=0) / len(sigma_list)
    retinex = gain * (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex))

    # Clip and convert the retinex image to uint8
    retinex = np.clip(retinex, 0, 255).astype(np.uint8)

    # Apply gamma correction
    dehazed_image = np.power(retinex / 255.0, 1.2) * 255.0
    dehazed_image = np.clip(dehazed_image, 0, 255).astype(np.uint8)

    return dehazed_image
from django.contrib.auth import logout
from django.contrib import messages

def logout_view(request):
    logout(request)
    messages.success(request, 'You have been successfully logged out.')

    return redirect('home')
def index(request):
    return render(request, 'dashboard.html')
def image(request):
    return render(request,'dehaze.html')
def video(request):
    return render(request,'video.html')
def home(request):
    return render(request,'home.html')
def log(request):
    return render(request,'login_signup.html')
def signup(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password = request.POST['password']
        user = CustomUser.objects.create_user(username=username, email=email, password=password)
        login(request, user)
        return redirect('signin')  # Redirect to the home page after successful sign-up
    return render(request, 'login_signup.html')

def signin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('index')  # Redirect to the home page after successful sign-in
    return render(request, 'login_signup.html')

def dehaze_video_view(request):
    if request.method == 'POST' and request.FILES.get('video'):
        video_file = request.FILES['video']

        # Save the uploaded video to a temporary location
        temp_video_path = 'temp_video.mp4'
        with open(temp_video_path, 'wb') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        # Specify the output path for the dehazed video
        output_video_path =r'C:\Users\afsal\PycharmProjects\imagedehyzeprg\imagedehyzeprg\hyzeapp\static\videos\hyzeappoutput_video_dehazed.mp4'

        # Perform video dehazing
        dehaze_video(temp_video_path, output_video_path)

        # Remove the temporary video file
        os.remove(temp_video_path)

        # Return the path to the dehazed video
        return JsonResponse({'processed_video_path': output_video_path})

    return JsonResponse({'error': 'No video file uploaded'})


def display_videos(request):

    dehazed_video_path =r'C:\Users\afsal\PycharmProjects\imagedehyzeprg\imagedehyzeprg\hyzeapp\static\videos\hyzeappoutput_video_dehazed.mp4'


    context = {

        'dehazed_video_path': dehazed_video_path,
    }

    return render(request, 'display.html', context)




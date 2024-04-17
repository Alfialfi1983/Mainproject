import cv2
import torch
import numpy as np
from tqdm import tqdm

def dehaze_frame(frame, model):
    # Convert frame to tensor
    frame_tensor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_tensor = (frame_tensor / 255.0)
    frame_tensor = torch.from_numpy(frame_tensor).float()
    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)

    # Dehaze the frame
    with torch.no_grad():
        dehazed_frame = model(frame_tensor)

    # Convert tensor back to numpy array
    dehazed_frame = (dehazed_frame.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)

    return dehazed_frame

def dehaze_video(video_path, output_path, skip_frames=0):
    # Open video capture
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Use MP4V codec for MP4 format
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load model on CPU
    model = torch.load(r'C:\Users\afsal\PycharmProjects\imagedehyzeprg\imagedehyzeprg\hyzeapp\snapshots\dehazer.pth', map_location=torch.device('cpu'))
    model.eval()

    progress_bar = tqdm(total=frame_count)
    frame_number = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            if frame_number % (skip_frames + 1) == 0:
                # Perform dehazing on the frame
                dehazed_frame = dehaze_frame(frame, model)

                # Display the dehazed frame
                cv2.imshow('Dehazed Video', dehazed_frame)
                cv2.waitKey(1)

                # Write the dehazed frame to the output video
                out.write(dehazed_frame)

                # Update progress bar
                progress_bar.update(1)

            frame_number += 1

        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    progress_bar.close()

    # Close OpenCV windows
    cv2.destroyAllWindows()

    return output_path

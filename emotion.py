import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from repvgg import create_RepVGG_A0 as create

# Load model (deployed architecture)
model = create(deploy=True)

# 8 Emotions
emotions = ("stressed", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise") 

def init(device):
    # Initialise model
    global dev
    dev = device
    model.to(device)
    # load with map_location to avoid device mismatch
    state = torch.load("./repvgg.pth", map_location=device)
    # handle whether state is a dict containing 'state_dict' or raw state_dict
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)

    cudnn.benchmark = True
    model.eval()


def detect_emotion(images, conf=True, temperature=2.0, return_probs=False):
    """
    images: list of numpy arrays (BGR as from OpenCV)
    conf: include confidence percent in returned label
    temperature: >1.0 softens (makes predictions less extreme), <1.0 sharpens
    """
    if len(images) == 0:
        return []

    with torch.no_grad():
        # Normalise and transform images
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        tensors = []
        for image in images:
            # Convert BGR (OpenCV) to RGB and ensure 3 channels
            try:
                if image is None or image.size == 0:
                    # create a dummy black image if invalid
                    pil = Image.new('RGB', (224, 224))
                else:
                    if image.ndim == 2:
                        # grayscale to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    # OpenCV BGR -> RGB
                    import cv2
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
            except Exception:
                # fallback
                pil = Image.fromarray(image)

            tensors.append(preprocess(pil))

        x = torch.stack(tensors)
        # Feed through the model
        outputs = model(x.to(dev))

        # Detect whether outputs are probabilities (some model forwards apply softmax)
        probs_mask = (outputs >= 0).all() and torch.allclose(outputs.sum(dim=1), torch.ones(outputs.size(0), device=outputs.device), atol=1e-3)
        if probs_mask:
            logits = torch.log(outputs.clamp(min=1e-12))
        else:
            logits = outputs

        # Apply temperature scaling then softmax for probabilities
        if temperature != 1.0:
            probs = F.softmax(logits / float(temperature), dim=1)
        else:
            probs = F.softmax(logits, dim=1)

        result = []
        for i in range(probs.size(0)):
            top_idx = int(torch.argmax(probs[i]).item())
            top_prob = float(probs[i][top_idx].item())
            label = emotions[top_idx]
            if conf:
                label_str = f"{label} ({top_prob*100:.1f}%)"
            else:
                label_str = label
            probs_list = [float(p) for p in probs[i].cpu().numpy()]
            if return_probs:
                result.append([label_str, top_idx, probs_list])
            else:
                result.append([label_str, top_idx])

    return result

from PIL import Image
import torchvision.transforms as T

def preprocess_image_pil(img: Image.Image):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor()
    ])
    return transform(img)

from PIL import Image
import numpy as np
from src.utils import preprocess_image_pil

def test_preprocess_image_shape():

    img = Image.fromarray(
        np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    )

    tensor = preprocess_image_pil(img)

    assert tensor.shape[0] == 3
    assert tensor.shape[1] == 224
    assert tensor.shape[2] == 224

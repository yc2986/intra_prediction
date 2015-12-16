import numpy as np
from PIL import Image

src = 'fig.npy'
img = np.load(src)

img = Image.fromarray(np.uint8(img))
img.save('out.png')
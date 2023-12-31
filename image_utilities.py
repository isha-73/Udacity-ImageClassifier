from PIL import Image
from data_utilities import test_transforms

mean, stdev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def processimage(path):
    
    with Image.open(path) as img:
        trans = test_transforms(mean,stdev)
        img = trans(img).numpy()

    return img
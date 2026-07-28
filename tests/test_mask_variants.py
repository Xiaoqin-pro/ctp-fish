import numpy as np
from PIL import Image
from datasets.mask_variants import apply_mask_variant

def test_foreground_and_background_views_are_complementary():
    image=Image.fromarray(np.full((2,2,3),200,dtype=np.uint8)); mask=Image.fromarray(np.array([[255,0],[0,255]],dtype=np.uint8))
    foreground=np.asarray(apply_mask_variant(image,mask,"foreground_only")); background=np.asarray(apply_mask_variant(image,mask,"background_only"))
    assert (foreground[0,1]==128).all() and (background[0,0]==128).all()

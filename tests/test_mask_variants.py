import numpy as np
from PIL import Image
from datasets.mask_variants import apply_mask_variant

def test_foreground_and_background_views_are_complementary():
    image=Image.fromarray(np.full((2,2,3),200,dtype=np.uint8)); mask=Image.fromarray(np.array([[255,0],[0,255]],dtype=np.uint8))
    foreground=np.asarray(apply_mask_variant(image,mask,"foreground_only")); background=np.asarray(apply_mask_variant(image,mask,"background_only"))
    assert (foreground[0,1]==128).all() and (background[0,0]==128).all()


def test_context_sanity_views_keep_shape_and_require_donor_when_needed():
    image=Image.fromarray(np.full((4,4,3),200,dtype=np.uint8)); mask=Image.fromarray(np.array([[255,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,255]],dtype=np.uint8))
    donor=Image.fromarray(np.array([[0,0,255,0],[0,0,0,0],[255,0,0,0],[0,0,0,0]],dtype=np.uint8))
    mask_only=np.asarray(apply_mask_variant(image,mask,"mask_only")); inpainted=np.asarray(apply_mask_variant(image,mask,"inpainted_background")); shuffled=np.asarray(apply_mask_variant(image,mask,"shuffled_mask_background",donor_mask=donor))
    assert mask_only.shape == inpainted.shape == shuffled.shape == (4,4,3)
    assert (mask_only[0,0] == 255).all() and (mask_only[0,1] == 0).all()
    assert (shuffled[0,2] == 128).all()

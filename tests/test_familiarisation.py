import pytest

from cued_sf2_lab.familiarisation import load_mat_img
from cued_sf2_lab.familiarisation import prep_cmap_array_plt, plot_image


def test_load_mat_img_1():
    """
    Test for an invalid image name.
    """
    img = 'wrong_img_name'
    img_info = 'X'
    cmap_info = {'map', 'map2'}
    with pytest.raises(ValueError):
        load_mat_img(img, img_info, cmap_info)

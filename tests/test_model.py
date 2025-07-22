from ..models.unet import build_unet

def test_model_compiles():
    model = build_unet()
    assert len(model.layers) > 0

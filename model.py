import timm

from cfenet import CFENet
from decoders.unet import SwinUNet

class SegmentationModel(nn.Module):
    """This class will only work properly if the encoder and decoder work correctly together already.
        This doesn't verify anything, just a binding class.
        
        The CFENet is designed only for 4 feature maps, so it will throw an error if more are given.    
    """
    def __init__(self, encoder, decoder, cfenet=False, encoder_channels=None):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        if not cfenet and encoder_channels is None:
            raise Exception("Pass the encoder_channels to use CFENet")

        self.cfenet = CFENet(encoder_channels) if cfenet else None
    
    def forward(self, x):
        features = self.encoder(x)
        permuted_features = [f.permute(0, 3, 1, 2) for f in features]

        if self.cfenet is not None:
            permuted_features = self.cfenet(permuted_features)

        decoder_output = self.decoder(permuted_features)
    
        return decoder_output


def build_model(config):
    model_cfg = config["model"]
    if model_cfg["encoder"] == "swin_t":
        encoder_channels = [96, 192, 384, 768]

    encoder = timm.create_model(
                model_cfg["timm_name"],
                pretrained=model_cfg.get("pretrained", True),
                features_only=True)
    
    if model_cfg["decoder"] == "unet":
        decoder = SwinUNet(encoder_channels)
    
    if model_cfg["cfenet"] not in [True, False]:
        raise ValueError("cfenet must be a bool")
    
    return SegmentationModel(encoder, decoder, model_cfg["cfenet"], encoder_channels)


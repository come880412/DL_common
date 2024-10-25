import timm

def build_model(args, device, num_classes):
    try:
        model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes).to(device)
        return model
    except:
        raise NameError("Model name error!")
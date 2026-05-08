from torchvision.transforms import v2


def augmentation(
    random_horizontal: float = 0.5,
    random_rotation_degrees: tuple[int, int] = (-90, 90),
) -> v2.Compose:
    """
    """
    return v2.Compose([
        v2.RandomHorizontalFlip(p=random_horizontal),
        v2.RandomRotation(degrees=random_rotation_degrees)
    ])
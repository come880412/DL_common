import argparse

def load_config():
    parser = argparse.ArgumentParser()

    # Path
    parser.add_argument('--data_dir', default='../dataset/train', help='path to data')
    parser.add_argument('--category_file', default="./dataset/category.json", help='Path to category.json')
    parser.add_argument('--model_dir', default='./checkpoints', help='path to model to save model!')
    parser.add_argument('--writer_path', default='./runs', help='Path to save record curve')
    parser.add_argument('--writer_overwrite', action="store_true", help='Whether to overwrite the writer folder')
    parser.add_argument('--resume', default=None, help='path to latest checkpoint (default: None)')

    # Inference parameters
    parser.add_argument('--output_path', default='./output', help='Path to prediction')
    parser.add_argument('--test_mode', default="val", help=["val", "test"])

    # Model
    parser.add_argument('--model_name', default="resnet18", help="Which model you want to use (default: resnet18)")

    # Leanring rate scheduler
    parser.add_argument("--warmup_epochs", type=int, default=3, help="Warmup epochs")
    parser.add_argument("--warmup_lr", type=float, default=1e-9, help="Warmup start learning rate")

    # Training hyperparameters
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=2, help="batch_size")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="gradient clipping")
    parser.add_argument('--n_cpus', type=int, default=4, help='number of cpu workers (default: 4)')
    parser.add_argument('--seed', type=int, default=2022, help='Fix random seed for reproduce (default: 2022)')

    # Data augmentation parameters
    parser.add_argument('--image_size', type=int, default=224, help='Image size (default: 224)')
    parser.add_argument('--brightness', type=float, default=0.4, help='Brightness value (default: 0.4)')
    parser.add_argument('--saturation', type=float, default=0.4, help='Saturation value (default: 0.4))')
    parser.add_argument('--contrast', type=float, default=0.4, help='Contrast value (default: 0.4)')
    parser.add_argument('--hue', type=float, default=0.1, help='Hue value (default: 0.1)')
    parser.add_argument('--rotate_deg', type=int, default=30, help='Rotation degree (default: 30)')
    parser.add_argument('--translate', type=tuple, default=(0.3, 0.3), help='Translate ratio (horizontal, vertical)')
    parser.add_argument('--scale', type=tuple, default=(0.5, 1.5), help='Scale ratio (min, max)')
    parser.add_argument('--flip', type=float, default=0.5, help='Probablity of flip (default: 0.5)')
    parser.add_argument('--erasing', type=float, default=0.1, help='Probablity of erasing (default: 0.1)')
    parser.add_argument('--mean', type=tuple, default=(0.4878, 0.4546, 0.4165), help='Dataset mean (R, G ,B)')
    parser.add_argument('--std', type=tuple, default=(0.2607, 0.2542, 0.2567), help='Dataset std (R, G ,B)')

    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate (default: 3e-4")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay parameter (default: 1e-4)")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument('--opt', default='adan', help=["sgd", "adan", "adamw"])

    args = parser.parse_args()

    return args
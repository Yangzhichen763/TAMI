from tqdm import tqdm
from basic.utils.io import glob_single_files, read_image_as_pil, save_feature, IMG_EXTENSIONS

try:
    from basic.utils.console.log import ColorPrefeb as CP
except:
    class CPType(type):
        def __getattr__(cls, item):
            return lambda x: x  # 返回恒等函数

    class CP(metaclass=CPType):
        pass

def extract_features_from_images(model_forward, input_dir, output_file, device='cpu'):
    # Initialize dictionary to store features
    all_features = {}

    # Process all images in the input directory
    image_paths = glob_single_files(input_dir, IMG_EXTENSIONS)

    pbar = tqdm(image_paths, desc=f"Processing images from {CP.path(input_dir)}", unit="image")
    for image_path in pbar:
        feats = model_forward(image_path)
        pbar.set_postfix({"Shape": list(feats.shape)})

        # Store features in the dictionary with image path as key
        all_features[image_path] = feats.cpu().numpy()

    # Save features to output file
    save_feature(all_features, output_file)
    print(f"All features saved to {CP.path(output_file)}")


def find_input_directory(input_root, until_keywords):
    # If no input-dir is specified, find the first directory that matches the until-keywords
    if until_keywords:
        for root, dirs, _ in os.walk(input_root):
            for dir_name in dirs:
                if any(keyword in dir_name for keyword in until_keywords):
                    yield os.path.join(root, dir_name)

if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from images using DINOv3 model.")
    parser.add_argument(
        '--input-root', '-r', type=str, default="~/Dataset/LLIE_dataset/",
        help="Directory containing images to process."
    )
    parser.add_argument(
        '--input-dir', '-i', type=str, default=None, # default="~/Dataset/LLIE_dataset/LOL_v2/Real_captured/",
        help="Directory containing images to process."
    )
    parser.add_argument(
        '--until-keywords', '-u', type=str, default=['high', 'low', 'GT', 'input'], nargs='*',  # Accept multiple keywords
        help="Keywords to match directory names under input-root."
    )
    parser.add_argument(
        '--output-root', '-o', type=str, default="  ~/Dataset/LLIE_feats/dinov3/",
        help="Directory to save the extracted features (e.g., .npy or .npz)."
    )
    parser.add_argument(
        '--gpus', '-g', type=str, default=None,
        help="Specify the GPU device(s) to use (e.g., '0' for GPU 0, '0,1' for GPUs 0 and 1)."
    )
    args = parser.parse_args()

    input_root = args.input_root
    input_directory = args.input_dir

    # Set device (GPU or CPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus if args.gpus else ""
    device = 'cuda' if args.gpus else 'cpu'

    # Load DINOv3 model
    from basic.archs.zoo.dinov3 import build_dinov3_model, forward_dinov3_model, build_transform
    model_name = "vit_small_patch16_dinov3.lvd1689m"
    model = build_dinov3_model(model_name)
    model.to(device)
    transform = build_transform()

    def model_forward(image_path):
        # Read image
        img_pil = read_image_as_pil(image_path)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)  # 1 × 3 × H × W and move it to the selected device

        # Extract features from the image
        feats = forward_dinov3_model(model, img_tensor)
        return feats

    def extract_feat_and_save(input_directory):
        # Prepare output file path
        output_rel_dir = os.path.relpath(input_directory, input_root)
        output_file = os.path.join(args.output_root, output_rel_dir)
        output_file = os.path.splitext(output_file.rstrip(os.sep))[0] + ".npz"

        if os.path.exists(output_file):
            print(f"Output file {CP.path(output_file)} already exists. Skipping.")
            return

        # Call the feature extraction function
        extract_features_from_images(model_forward, input_directory, output_file, device)

    # If input-dir is not provided, search for it under input-root using until-keywords
    if input_directory is not None:
        extract_feat_and_save(input_directory)
        exit(0)
    if not args.until_keywords:
        print("No input-dir or until-keywords specified. Exiting.")
        exit(1)  # Exit if no input-dir or until-keywords are provided
    input_directories = list(find_input_directory(input_root, args.until_keywords))
    if not input_directories:
        print(f"No directories found matching keywords {args.until_keywords}. Exiting.")
        exit(1)  # Exit if no directory matches the keywords
    print(f"Found {len(input_directories)} directories matching keywords {args.until_keywords}: {', '.join([CP.path(d) for d in input_directories])}\n")
    for input_directory in input_directories:
        extract_feat_and_save(input_directory)



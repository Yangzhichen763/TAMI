import os
import numpy as np
import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
from basic.utils.io import glob_single_files, save_feature, IMG_EXTENSIONS

try:
    from basic.utils.console.log import get_root_logger
    logger = get_root_logger(force_set_info=True)
    def print(*args, **kwargs):
        logger.info(*args, **kwargs)
except ImportError:
    from builtins import print as original_print
    def print(*args, **kwargs):
        original_print(*args, **kwargs)

try:
    from basic.utils.console.log import ColorPrefeb as CP
except:
    class CPType(type):
        def __getattr__(cls, item):
            return lambda x: x  # 返回恒等函数

    class CP(metaclass=CPType):
        pass

def load_features_from_npz(folder_path, include_keywords=None, include_dirs=None):
    """
    Loads all .npz files from a folder and returns a list of features.
    Each .npz file contains a dictionary, and we're interested in extracting the feature arrays.
    """
    features = []
    file_names = []
    all_file_names = [file_name for file_name in glob_single_files(folder_path, ['npz'])]
    all_file_names = [
        f for f in all_file_names
        if not (
                include_dirs is not None and not any(dir_name in f for dir_name in include_dirs) or
                include_keywords is not None and not any(keyword in f for keyword in include_keywords)
        )
    ]

    for file_path in tqdm(all_file_names, desc="Loading NPZ files"):

        data = np.load(file_path)

        if len(data) == 0:
            continue
        feature_data = np.vstack(list(data.values()))
        # feature_data = np.vstack(list(data.values())[:4])
        features.append(feature_data)
        file_names.append(file_path)

    common_prefix = os.path.commonprefix(file_names)
    file_names = [file_name.replace(common_prefix, '').replace('.npz', '') for file_name in file_names]
    return features, file_names

def load_reduced_features(feat_save_path):
    """
    Loads the reduced features from a .npz file.
    """
    data = np.load(feat_save_path)
    features = []
    file_names = []
    for f in data.keys():
        features.append(data[f])
        file_names.append(f)
    return features, file_names

def reduce_dimensions(features, n_components=2):
    """
    Reduces the dimensionality of the features using UMAP.
    """
    _features = [f.reshape(-1, f.shape[-1]) for f in features]
    all_features = np.vstack(_features)

    # Apply UMAP dimensionality reduction
    reducer = umap.UMAP(n_components=n_components, verbose=True, n_jobs=-1) # 使用 random_state=42 就不能使用并行计算了
    reduced_features = reducer.fit_transform(all_features)

    _reduced_features = np.split(reduced_features, np.cumsum([f.shape[0] for f in _features])[:-1])
    return _reduced_features

def reduce_dimensions_gpu(features, n_components=2):
    """
    Reduces the dimensionality of the features using UMAP with GPU support.
    """
    from cuml.manifold import UMAP as cuUMAP    # conda install -c rapidsai -c nvidia -c conda-forge cuml=23.02 python=3.8 cudatoolkit=11.6

    _features = [f.reshape(-1, f.shape[-1]) for f in features]
    all_features = np.vstack(_features)

    # Convert data to GPU arrays using CuPy
    all_features_gpu = cp.array(all_features)

    # Apply UMAP with GPU
    reducer = cuUMAP(n_components=n_components, random_state=42)
    reduced_features_gpu = reducer.fit_transform(all_features_gpu)

    reduced_features = cp.asnumpy(reduced_features_gpu)
    _reduced_features = np.split(reduced_features, np.cumsum([f.shape[0] for f in _features])[:-1])

    return _reduced_features

def plot_reduced_features(
        reduced_features, file_names, save_path=None,
):
    """
    Plots the reduced features on a 2D plot.
    Each point is labeled by its corresponding file name.
    """
    plt.figure(figsize=(10, 10))

    for i, (_reduced_features, file_name) in enumerate(zip(reduced_features, file_names)):
        # num_sample = min(sample_max_size, int(_reduced_features.shape[0] * sample_ratio))
        # indices = np.random.choice(_reduced_features.shape[0], num_sample, replace=False)

        plt.scatter(_reduced_features[:, 0], _reduced_features[:, 1], s=5, alpha=0.6, edgecolor='none', label=file_name)

    plt.legend(fontsize=8)
    plt.axis('off')

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def save_reduced_features(
        features, file_names, save_path=None,
):
    # Initialize dictionary to store features
    all_features = {}

    for file_name, feature in zip(file_names, features):
        all_features[file_name] = feature

    # Save features to output file
    save_feature(all_features, save_path)
    print(f"All features saved to {CP.path(save_path)}")


if __name__ == "__main__":
    import os
    import argparse

    parser = argparse.ArgumentParser(description="Dimensionality reduction of features using UMAP.")
    parser.add_argument(
        '--gpus', '-g', type=str, default=None,
        help="Specify the GPU device(s) to use (e.g., '0' for GPU 0, '0,1' for GPUs 0 and 1)."
    )
    args = parser.parse_args()


    # Load features
    folder_path = "~/Dataset/LLIE_feats/dinov3/"  # Specify the folder containing .npz files
    include_keywords = ['high', 'low'] # ['high', 'GT']
    include_dirs = ['LOL_v2'] # ['LOL_v1', 'LOL_v2', 'SDSD_indoor_png', 'SDSD_outdoor_png']
    features, file_names = load_features_from_npz(folder_path, include_keywords, include_dirs)

    # Reduce dimensionality
    num_total_samples = sum([f.shape[0] * f.shape[1] for f in features])
    meter_str = '0'
    if num_total_samples < 1000:
        meter_str = f"{num_total_samples}"
    elif num_total_samples < 1e6:
        meter_str = f"{num_total_samples / 1e3:.2f}K"
    elif num_total_samples < 1e9:
        meter_str = f"{num_total_samples / 1e6:.2f}M"
    else:
        meter_str = f"{num_total_samples / 1e9:.2f}B"
    print(f"Reducing dimensionality of {CP.number(meter_str)} samples...")
    if args.gpus is not None:
        import cupy as cp   # conda install -c conda-forge cupy cudatoolkit=11.6, or pip install cupy-cuda116
        device_id = args.gpus
        with cp.cuda.Device(device_id):
            reduced_features = reduce_dimensions_gpu(features, n_components=2)
    else:
        reduced_features = reduce_dimensions(features, n_components=2)
    print(f"Finished reducing dimensionality.")

    # ====================================
    #  save and plot the reduced features
    # ====================================
    from basic.utils.console.log import get_striped_time_str
    save_dir = "./.plotlogs/feat_reduction/"
    os.makedirs(save_dir, exist_ok=True)
    time_stamp = get_striped_time_str()
    feat_save_path = os.path.join(save_dir, f"UMAP_visualization_{time_stamp}.npz")

    print(f"Saving reduced features to {CP.path(feat_save_path)}...")
    save_reduced_features(reduced_features, file_names, save_path=feat_save_path)
    loaded_features, loaded_file_names = load_reduced_features(feat_save_path)
    print(f"Saved {CP.number(len(loaded_features))} features to {CP.path(feat_save_path)}.")

    # Plot the reduced features
    print(f"Plotting reduced features...")
    plot_save_path = feat_save_path.replace(".npz", ".png")
    plot_reduced_features(reduced_features, file_names, plot_save_path)
    print(f"Finished plotting reduced features.")

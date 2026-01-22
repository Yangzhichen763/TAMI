import os
from typing import List, Optional

from einops import rearrange
import os.path as osp
import numpy as np
import cv2
import functools

from torchvision import models
import torch
import matplotlib
from matplotlib.patches import FancyArrowPatch, Circle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from basic.utils.console.log import get_striped_time_str


try:
    from basic.utils.console.log import ColorPrefeb as CP
except:
    class CPType(type):
        def __getattr__(cls, item):
            return lambda x: x  # 返回恒等函数

    class CP(metaclass=CPType):
        pass


# 如果使用到了多线程绘制 matplotlib 图，则需要设置 matplotlib.use('Agg')
try:
    from basic.utils.parallelNsync.singleton_thread import thread_parallel
    matplotlib.set_loglevel("warning")
    matplotlib.use('Agg')
except:
    def thread_parallel(priority=0):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)
            return wrapper
        return decorator


import matplotlib.pyplot as plt

#region ==[Plotter]
initialized_plotter = {}


'''
和绘图相关的工具函数
'''


def detach_all(func):
    def _detach_all(obj, _seen=None):
        if _seen is None:
            _seen = set()

        # 防止循环引用
        obj_id = id(obj)
        if obj_id in _seen:
            return obj
        _seen.add(obj_id)

        try:
            if obj is None:
                return obj
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu()
            elif isinstance(obj, tuple):
                return tuple(_detach_all(a) for a in obj)
            elif isinstance(obj, list):
                return [_detach_all(a) for a in obj]
            elif isinstance(obj, dict):
                return {k: _detach_all(v) for k, v in obj.items()}
            else:
                return obj
        except Exception:
            return obj

    @functools.wraps(func)
    def wrapper(*args, **kwargs):

        args = [_detach_all(arg) for arg in args]
        kwargs = {k: _detach_all(v) for k, v in kwargs.items()}

        return func(*args, **kwargs)
    return wrapper

def save_only_method(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self._save_only:
            self.save_action(func.__name__, *args, **kwargs)
            return None
        return func(self, *args, **kwargs)
    return wrapper


def get_root_plotter(
        plotter_name='main', plotter_class='Plotter',
        plot_root=None, plot_sub_dir=None,
        with_timestamp=True,
):
    from os import path as osp

    plotter_name = f"{plotter_name}_{plotter_class}"

    # [pre-]
    plotter = initialized_plotter.get(plotter_name, None)
    # if the plotter has been initialized, just return it
    if plotter_name in initialized_plotter:
        return plotter

    # [get file name]
    plot_dir = plot_root or "./.plotlogs"
    if plot_sub_dir is not None:
        if with_timestamp:
            plot_sub_dir = f"{plot_sub_dir}_{get_striped_time_str()}"
        plot_dir = osp.join(plot_dir, plot_sub_dir)
    if not osp.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    plotter = globals()[plotter_class](plot_dir)
    initialized_plotter[plotter_name] = plotter
    return plotter


from basic.utils.convert import tensor2numpy


# noinspection SpellCheckingInspection
class Plotter:
    def __init__(self, root, dpi=256, gap=0.02):
        """
        Args:
            root (None or str): Root directory for saving plots.
            dpi (int): Dots per inch for the plot. Default is 256.
            gap (float): Gap between subplots. Default is 0.02.
        """
        if root is None:
            root = "./.plotlogs"

        self.root = root
        self.dpi = dpi
        self.gap = gap

        self._save_only = False

        self.reduce_params = {
            'umap': dict(
                # metric='cosine',
                random_state=42,
            ),
            'tsne': dict(
                n_jobs=1,
                random_state=42,
            ),
            'pca': dict(),
        }
        self.reduce_params.update({
            't-sne': self.reduce_params['tsne'],
        })

    def set_save_only(self, save_only=True):
        self._save_only = save_only


    @staticmethod
    def _remove_axes(ax):
        """
        Remove axes from a matplotlib figure.
        Args:
            ax (Any): Axes to remove.
        """
        from matplotlib.ticker import NullFormatter
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    @staticmethod
    def _subplots(nrows, ncols, figsize=(12, 9)):
        """
        Create subplots with a given number of rows and columns.
        Thread-safe version using Figure and FigureCanvasAgg instead of pyplot.

        Args:
            nrows (int): Number of rows in the subplot grid.
            ncols (int): Number of columns in the subplot grid.
            figsize (tuple[float, float] or None): Figure size as (width, height) in inches. Default is (12, 9).

        Returns:
            tuple: (fig, axes) where fig is the figure object and axes is a 2D array of axes objects.
        """
        # 使用独立的 Figure 对象，避免 pyplot 全局状态，实现线程安全
        fig = Figure(figsize=figsize)
        FigureCanvasAgg(fig)  # 创建 canvas，但不保存引用（会被 figure 管理）

        # 始终返回 2D 数组，确保 axes[i, j] 索引能正常工作
        if nrows == 1 and ncols == 1:
            axes = np.array([[fig.add_subplot(111)]])
        elif nrows == 1:
            axes_1d = [fig.add_subplot(1, ncols, i+1) for i in range(ncols)]
            axes = np.array([axes_1d])
        elif ncols == 1:
            axes = np.array([[fig.add_subplot(nrows, 1, i+1)] for i in range(nrows)])
        else:
            axes = np.array([[fig.add_subplot(nrows, ncols, i*ncols + j + 1)
                            for j in range(ncols)] for i in range(nrows)])

        return fig, axes

    def _append_fig_name(self, fig_name: str, suffix: str) -> str:
        if fig_name is not None:
            if fig_name.endswith('+'):
                fig_name = fig_name[:-1]
                fig_name = f"{fig_name}_{suffix}+"
            else:
                fig_name = f"{fig_name}_{suffix}"
        else:
            fig_name = suffix
        return fig_name

    def _validate_fig_name(self, fig_name: str, suffix: str='png') -> str:
        if fig_name is not None:
            if fig_name.endswith('+'):
                fig_name = f"{fig_name[:-1]}_{get_striped_time_str()}.{suffix}"
            else:
                fig_name = f"{fig_name}.{suffix}"
        else:
            fig_name = f"{get_striped_time_str()}.{suffix}"
        file_path = osp.join(self.root, fig_name)
        dir_path = osp.dirname(file_path)
        if not osp.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        return file_path

    def _grayscale_to_heatmap(self, gray_tensor, cmap='viridis', normalize=True):
        """
        Convert a grayscale image tensor to a heatmap using a colormap.

        Args:
            gray_tensor (torch.Tensor): (H, W) or (1, H, W)
            cmap (str): Matplotlib colormap (default: 'viridis').
            normalize (bool): Whether to normalize the grayscale values to [0, 1] (default: True).

        Returns:
            np.ndarray: (H, W, 3)
        """

        if gray_tensor.ndim == 3:
            gray_tensor = gray_tensor.squeeze(0)
        assert gray_tensor.ndim == 2, f"gray_tensor must be (H, W) or (1, H, W), but got {gray_tensor.shape}"

        if normalize:
            gray_tensor = (gray_tensor - gray_tensor.min()) / (gray_tensor.max() - gray_tensor.min())

        colormap = cm.get_cmap(cmap)
        heatmap = colormap(gray_tensor)

        heatmap = (heatmap * 255).astype(np.uint8)[..., :3]
        return heatmap

    def _reduce_and_visualize(
            self, features, *joint_features,
            fig_name: str=None, reduce_method: str= 'pca',
            gradient: bool=False, cmap: str=None,
            scatter_size=10
    ):
        """
        Reduce the feature dimension and visualize the results.

        Args:
            features (list[torch.Tensor]): List of feature tensors (C,)
                features 的不同 list 会显示为不同的颜色，如果要显示为同一个颜色，可以传入只有 一个 tensor 的 list，
            fig_name (str, optional): Figure name (default: None).
            reduce_method (str, optional): Dimensionality reduction method ('umap', 'tsne', 'pca') (default: 'pca').
        """
        if reduce_method not in self.reduce_params:
            raise ValueError(f"Unsupported method: {reduce_method}. Supported methods are: {list(self.reduce_params.keys())}")

        all_features = [features] + [*joint_features]
        n_b = len(all_features)
        n_levels = len(features)
        if cmap is not None:
            if isinstance(cmap, list):
                cmaps = cmap
            else:
                cmaps = [cmap] * n_b
        else:
            # cmaps = ['Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'Greys']
            if n_b == 1:
                if gradient:
                    cmaps = ['viridis_r']
                else:
                    cmaps = ['tab10']
            else:
                cmaps = generate_tab10_saturation_gradients(n_levels=n_levels + 1)

        # ==[reduce]==
        cated_all_features = [x for f in all_features for x in f]
        all_reduced_feats_2d, _ = reduce_dimensionality(
            cated_all_features, dim=2, method=reduce_method, normalize=False,
            **self.reduce_params[reduce_method]
        )  # pca outputs a list of reduced tensors

        # ==[plot]==
        fig, axes = self._subplots(1, 1, figsize=(10, 10))
        ax = axes[0, 0]
        acc_len = 0
        for b, all_features_b in enumerate(all_features):   # (B, N, T, dim)
            if b == n_b - 1:
                reduced_feats_2d = all_reduced_feats_2d[acc_len:]
            else:
                reduced_feats_2d = all_reduced_feats_2d[acc_len:acc_len + len(all_features_b)]
            cmap = cmaps[b % len(cmaps)]
            acc_len += len(all_features_b)

            # reshape
            data_2d, joint_data_2d = reduced_feats_2d[0], reduced_feats_2d[1:]
            data_2d = data_2d.reshape(2, -1).T
            joint_data_2d = [d.reshape(2, -1).T for d in joint_data_2d]

            # Plot scatter points
            if gradient:
                all_datas = torch.stack([data_2d] + joint_data_2d, dim=0)   # (N, T, dim), dim=2
                if isinstance(cmap, str) or isinstance(cmap, mcolors.Colormap):
                    colormap = matplotlib.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
                    colors = colormap(np.linspace(0, 1, len(all_datas)))
                elif isinstance(cmap, list) or isinstance(cmap, np.ndarray):
                    colors = cmap
                else:
                    raise ValueError(f"Unsupported cmap. Supported types are: str, matplotlib.colors.Colormap, list. But got {type(cmap)}")

                # plot point
                for t, d_t in enumerate(all_datas):
                    ax.scatter(d_t[:, 0], d_t[:, 1], color=colors[(t + 1) % len(colors)], s=1, alpha=1, marker='o', zorder=10)
                # plot intra-lines
                for n, d_n in enumerate(all_datas.permute(1, 0, 2)):
                    for t in range(len(all_datas) - 1):
                        ax.plot([d_n[t, 0], d_n[t+1, 0]], [d_n[t, 1], d_n[t+1, 1]], color=colors[(t + 1) % len(colors)], linewidth=1, alpha=1, zorder=5)
                # plot inter-lines
                if b == 0:
                    mean_point_A_list = []
                    d_i_radius_A_list = []
                mean_point_B_list = []
                d_i_radius_B_list = []
                for n, d_n in enumerate(all_datas.permute(1, 0, 2)):
                    mean_point_B = d_n.mean(dim=0)                              # 计算平均值点
                    d_i_radius_B = (d_n - mean_point_B).norm(dim=1).max()       # 计算最小包围圆半径
                    d_i_radius_B = d_i_radius_B * 1.1
                    def distance_to_scale(x):
                        return x * 10 # x * self.dpi / 72 # (x * 72 / self.dpi) ** 2 * math.pi
                    if b != 0:
                        # 绘制箭头
                        shrink_A = distance_to_scale(d_i_radius_A_list[n])
                        shrink_B = distance_to_scale(d_i_radius_B)
                        arrow_line = FancyArrowPatch(
                            mean_point_A_list[n], mean_point_B,
                            arrowstyle="-",
                            linestyle="--",     # 虚线连接线
                            shrinkA=shrink_A,
                            shrinkB=shrink_B,
                            linewidth=0.5,
                            mutation_scale=1,
                            color=colors[1],
                            alpha=1,
                            zorder=1
                        )
                        arrow_head = FancyArrowPatch(
                            mean_point_A_list[n], mean_point_B,
                            arrowstyle="-|>",   # 实线箭头
                            shrinkA=shrink_A,
                            shrinkB=shrink_B,
                            linewidth=0,
                            mutation_scale=8,  # 箭头大小
                            color=colors[1],
                            alpha=1,
                            zorder=1
                        )
                        ax.add_patch(arrow_line)
                        ax.add_patch(arrow_head)

                    mean_point_B_list.append(mean_point_B)
                    d_i_radius_B_list.append(d_i_radius_B)
                    # 绘制圆形区域
                    circle = Circle(
                        mean_point_B, d_i_radius_B,
                        color=colors[0], fill=True, linewidth=0, alpha=0.5, zorder=2,
                    )
                    ax.add_patch(circle)
                    # scale = distance_to_scale(d_i_radius_B) ** 2 * math.pi
                    # ax.scatter(
                    #     mean_point_B[0], mean_point_B[1],
                    #     s=scale,
                    #     color=colors[0], edgecolors='none', alpha=1, zorder=2
                    # )
                mean_point_A_list = mean_point_B_list
                d_i_radius_A_list = d_i_radius_B_list
            else:
                if isinstance(cmap, str) or isinstance(cmap, mcolors.Colormap):
                    colormap = matplotlib.colormaps.get_cmap(cmap) if isinstance(cmap, str) else cmap
                    colors = colormap.colors
                elif isinstance(cmap, list) or isinstance(cmap, np.ndarray):
                    colors = cmap
                else:
                    raise ValueError(f"Unsupported cmap: {cmap}. Supported types are: str, colors.Colormap, list.")

                for t, d_t in enumerate(joint_data_2d):
                    ax.scatter(d_t[:, 0], d_t[:, 1], color=colors[(t + 1) % len(colors)], s=scatter_size, alpha=1, marker='o', zorder=9)
                ax.scatter(data_2d[:, 0], data_2d[:, 1], color='red', s=scatter_size, alpha=1, marker='*', zorder=10)

        # plot range
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_range = max(xlim[1], -xlim[0])
        y_range = max(ylim[1], -ylim[0])
        max_range = max(x_range, y_range)
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)

        # plot style
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
        ax.set_aspect('equal')

        # legend
        ax.legend(
            labels=[f'{i+1}' for i in range(n_levels)],
            loc='best',
            fontsize='small',
            frameon=True,
        )

        # save plot
        self._grid_save(fig_name, fig=fig)
        fig.clear()

    def _grid_save(self, fig_name=None, settings='image', fig=None):
        """Save the figure with tight layout and adjusted spacing.

        Args:
            fig_name: Name of the figure file
            settings: 'image' or 'graph' style settings
            fig: Figure object to save. If None, will try to get from plt.gcf() (not thread-safe)
        """
        if fig is None:
            # Fallback to pyplot for backward compatibility, but not thread-safe
            fig = plt.gcf()
        ax = fig.gca() if fig.axes else None

        # 以下部分连续的代码的目的是为了在 Jupyter Notebook 中（在深色背景下）能够显示正常
        fig.patch.set_facecolor('none')
        for ax in fig.axes:
            ax.set_facecolor('none')
        fig.patch.set_alpha(1.0)

        if settings == 'image':
            # fig.tight_layout()
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=self.gap, hspace=self.gap)
        elif settings == 'graph':
            grid_color = 'gray'
            ax.tick_params(which='minor', length=0)
            ax.tick_params(which='major', length=0)
            # Set spines (borders)
            for spine in ax.spines.values():
                spine.set_linewidth(1)
                spine.set_color(grid_color)
            # Add grid lines for both major and minor ticks (light gray)
            ax.grid(True, which='major', linestyle=(0, (6, 3)), linewidth=0.5, alpha=1.0, color=grid_color, zorder=-10)
            ax.grid(True, which='minor', linestyle=(0, (6, 3)), linewidth=0.5, alpha=0.3, color=grid_color, zorder=-10)
            # Set y-axis grid (same as before)
            ax.grid(True, which='both', axis='y', linestyle=(0, (6, 3)), alpha=1.0, color=grid_color, zorder=-10)

        # [save figure]
        file_path = self._validate_fig_name(fig_name)
        fig.savefig(file_path, dpi=self.dpi, bbox_inches='tight', pad_inches=0)

    def _show_image_mode(self, ax, image, cmap, mode: str='normalize', ref=None):
        if mode == 'normalize':
            from matplotlib.colors import Normalize

            if ref is not None:
                vmin = ref.min().item()
                vmax = ref.max().item()
            else:
                vmin = image.min().item()
                vmax = image.max().item()
            norm = Normalize(vmin=vmin, vmax=vmax)

            im = ax.imshow(image, cmap=cmap, norm=norm)
        elif mode == 'slope_norm':
            from matplotlib.colors import TwoSlopeNorm

            if ref is not None:
                vmax = torch.abs(ref).max().item()
            else:
                vmax = torch.abs(image).max().item()
            vmin = -vmax

            norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

            im = ax.imshow(image, cmap=cmap, norm=norm)
        else:
            raise ValueError(f'Unsupported mode: {mode}. Supported modes are: normalize, tone.')
        return im


    def _cv_save(self, image: np.ndarray, fig_name=None):
        file_path = self._validate_fig_name(fig_name)
        cv2.imwrite(file_path, image)


    @thread_parallel()
    def save_action(self, func_name, *args, **kwargs):
        import numpy as np
        import os

        func_dict = {
            'func_name': func_name,
            'args': args,
            'kwargs': kwargs,
        }
        file_path = self._validate_fig_name(kwargs.get('fig_name', None), suffix='npz')

        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        np.savez_compressed(file_path, func_dict=func_dict)

    def parse_action(self, path, any_original_func=False):
        import numpy as np
        import os

        if not os.path.exists(path):
            raise FileNotFoundError(f'File not found: "{path}"')

        loaded_data = np.load(path, allow_pickle=True)

        func_dict = loaded_data['func_dict'].item() if hasattr(loaded_data['func_dict'], 'item') else loaded_data['func_dict']

        func_name = str(func_dict['func_name'])
        args = func_dict['args']
        kwargs = func_dict['kwargs']

        if not isinstance(args, tuple):
            args = tuple(args) if isinstance(args, (list, np.ndarray)) else (args,)
        if not isinstance(kwargs, dict):
            kwargs = dict(kwargs) if hasattr(kwargs, '__iter__') and not isinstance(kwargs, str) else {}

        func = getattr(self, func_name)
        if any_original_func:
            from basic.utils.general import get_original_callable
            return get_original_callable(func)(*args, **kwargs)
        return func(*args, **kwargs)

    @detach_all
    @thread_parallel()
    def delete(self, file_name):
        file_path = osp.join(self.root, file_name)
        if os.path.exists(file_path):
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                import shutil
                shutil.rmtree(file_path)

    # noinspection PyUnresolvedReferences
    @detach_all
    @save_only_method
    @thread_parallel()
    def heatmap(
            self, data, fig_name=None,
            mode='normalize', auto_grid=True,
            cmap='viridis', rearrange_option=None,
            show_colorbar=True, colorbar_kwargs=None,
            **rearrange_kwargs
    ):
        """
        输入特征图，使用热力图的方式将通道方向的所有图可视化出来：(B, C, H, W) -> B×C 个 (H, W)

        使用指南：
        * 输入 data 为特征，形状为 (B, C, H, W)
        * fig_name：data 保存为的可视化特征图的名称，最后如果带有 '+' 则会追加时间戳（比如 'feat+'），如果不带有 '+' 会默认覆盖同名文件

        Args:
            data (torch.Tensor): (H, W), (C, H, W) or (B, C, H, W)
            fig_name (str): Figure name (default: None).
            mode (str): 'normalize', 'slope_norm' (default: 'normalize').
            auto_grid (bool): If True, arrange B×C images into a near-square grid layout (default: True).
            cmap (str): Matplotlib colormap (default: 'viridis').
            rearrange_option (str, optional): Einops-style rearrangement string.
            **rearrange_kwargs: Additional arguments for `rearrange`.
        """
        if rearrange_option is not None:
            data = rearrange(data, rearrange_option, **rearrange_kwargs)

        if data.ndim == 1:
            rows, cols = 1, 1
            data = data.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        elif data.ndim == 2:
            rows, cols = 1, 1
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndim == 3:
            rows, cols = 1, data.shape[0]
            data = data.unsqueeze(0)
        elif data.ndim == 4:
            rows, cols = data.shape[0], data.shape[1]
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

        # [auto grid layout]
        original_total = rows * cols  # 保存原始图像数量
        if auto_grid and rows >= 1 and cols >= 1:
            total = rows * cols
            # 目标：找到 rows_new 和 cols_new，使得 rows_new × cols_new >= total 且接近方形
            sqrt_total = int(np.sqrt(total))

            best_ratio = float('inf')
            best_rows = rows
            best_cols = cols

            for test_rows in range(sqrt_total, total + 1):
                test_cols = (total + test_rows - 1) // test_rows
                if test_rows * test_cols < total:
                    continue
                ratio = test_cols / test_rows
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_rows = test_rows
                    best_cols = test_cols
                    if 1 <= ratio <= 1.6:
                        break

            data_flat = data.reshape(-1, *data.shape[-2:])  # (total, H, W)
            # 如果 best_rows * best_cols > total，需要填充空位，最后一个图像作为填充（后面会隐藏）
            if best_rows * best_cols > total:
                padding = best_rows * best_cols - total
                last_image = data_flat[-1:].expand(padding, -1, -1)
                data_flat = torch.cat([data_flat, last_image], dim=0)

            data = data_flat.reshape(best_rows, best_cols, *data.shape[-2:])
            rows, cols = best_rows, best_cols

        h, w = data.shape[-2:]
        base_pixel = 50
        while True:
            width, height = cols + self.gap * (cols - 1), rows + self.gap * (rows - 1)
            width, height = width * (base_pixel * w / self.dpi), height * (base_pixel * h / self.dpi)
            if (width < 4320 / self.dpi and height < 4320 / self.dpi) or base_pixel <= 1:
                break
            base_pixel = int(base_pixel * 0.95)
        fig, axes = self._subplots(rows, cols, figsize=(width, height))

        # [plot images]
        image_idx = 0
        for i in range(rows):
            for j in range(cols):
                if image_idx < original_total:
                    image = data[i, j]
                    ax = axes[i, j]

                    if image.shape[-2:] == (1, 1):
                        ref_data = data
                    else:
                        ref_data = image

                    self._remove_axes(ax)
                    self._show_image_mode(ax, image, cmap=cmap, mode=mode, ref=ref_data)
                else:
                    ax = axes[i, j]
                    ax.axis('off')

                image_idx += 1

        if show_colorbar:
            norm = None
            if mode == 'normalize':
                norm = mcolors.Normalize(vmin=float(data.min()), vmax=float(data.max()))
            elif mode == 'slope_norm':
                norm = mcolors.Normalize(vmin=float(data.min()), vmax=float(data.max()))

            mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])

            if colorbar_kwargs is None:
                colorbar_kwargs = {}

            fig.colorbar(
                mappable,
                ax=axes,
                location='right',
                **colorbar_kwargs
            )

        self._grid_save(fig_name, fig=fig)
        fig.clear()

    # noinspection PyUnresolvedReferences
    @detach_all
    @save_only_method
    @thread_parallel()
    def image(self, data, fig_name=None, rearrange_option=None, save_as_single=True, **rearrange_kwargs):
        """
        网格格式拼接展示延 BatchSize 方向上的所有图像：(B, C, H, W) -> B 个 (C, H, W)；
            其中：要满足 data 是 RGB 图像或者灰度图；数据的范围是 [0, 1]

        使用指南：
        * 输入 data 为图像，形状为 (B, C, H, W)
        * fig_name：data 保存为的可视化图像的名称，最后如果带有 '+' 则会追加时间戳（比如 'image+'），如果不带有 '+' 会默认覆盖同名文件

        Args:
            data (torch.Tensor): (H, W), (C, H, W) or (B, C, H, W)
            fig_name (str): Figure name (default: None).
            rearrange_option (str, optional): Einops-style rearrangement string.
            **rearrange_kwargs: Additional arguments for `rearrange`.
        """
        if rearrange_option is not None:
            data = rearrange(data, rearrange_option, **rearrange_kwargs)

        if data.ndim == 2:
            num = 1
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndim == 3:
            num = 1
            data = data.unsqueeze(0)
        elif data.ndim == 4:
            num = data.shape[0]
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

        h, w = data.shape[-2:]
        if save_as_single:
            for i in range(num):
                # 每张图单独的 fig 大小
                figsize = (w / self.dpi, h / self.dpi)
                fig, axex = self._subplots(1, 1, figsize=figsize)
                ax = axex[0, 0]

                image = tensor2numpy(data[i], reverse_channels=False, change_range=False)

                self._remove_axes(ax)
                ax.axis('off')

                if image.ndim == 2 or image.shape[-1] == 1:
                    ax.imshow(image, cmap='gray')
                else:
                    ax.imshow(image)

                # 保存
                if fig_name is None:
                    save_name = f"image_{i}"
                else:
                    if fig_name.endswith("+"):
                        base = fig_name[:-1]
                        save_name = f"{base}_{i}+"
                    else:
                        save_name = f"{fig_name}_{i}"

                self._grid_save(save_name, fig=fig)
                fig.clear()
        else:
            # 图像横着堆叠
            width, height = num + self.gap * (num - 1), 1
            width, height = width * (w / self.dpi), height * (h / self.dpi)
            fig, axes = self._subplots(1, num, figsize=(width, height))

            # [plot images]
            for i in range(num):
                image = data[i]
                ax = axes[0, i]

                self._remove_axes(ax)
                ax.axis('off')

                image = tensor2numpy(image, reverse_channels=False)
                if image.ndim == 2 or image.shape[-1] == 1:
                    ax.imshow(image, cmap='gray')
                else:
                    ax.imshow(image)

            self._grid_save(fig_name, fig=fig)
            fig.clear()

    # noinspection PyUnresolvedReferences
    @detach_all
    @save_only_method
    @thread_parallel()
    def semantic_feature_map(
            self,
            data, fig_name=None,
            reduce_method="tsne",

            # image plot options
            image_plot=True,

            # scatter plot options
            scatter_plot=False, scatter_fig_alias="2d+",

            rearrange_option=None, **rearrange_kwargs
    ):
        """
        网格格式拼接展示延 BatchSize 和 timeline 方向上的所有特征图像：(B, T, C, H, W) -> B×T 个 (C, H, W)，展示时以 RGB 图展示
        使用 pca 方式降维特征

        使用指南：
        * 输入 data 为潜空间特征，形状为 (B, C, H, W)
        * fig_name：data 保存为的可视化图像的名称，最后如果带有 '+' 则会追加时间戳（比如 'image+'），如果不带有 '+' 会默认覆盖同名文件
        * scatter_plot：是否额外显示二维散点图，使用PCA维度作为颜色

        Args:
            data (torch.Tensor): (C, H, W), (B, C, H, W) or (B, T, C, H, W)
            fig_name (str): Figure name (default: None).
            rearrange_option (str, optional): Einops-style rearrangement string.
            scatter_plot (bool, optional): Whether to show additional scatter plot.
            **rearrange_kwargs: Additional arguments for `rearrange`.
        """
        if rearrange_option is not None:
            data = rearrange(data, rearrange_option, **rearrange_kwargs)

        if data.ndim == 3:
            rows, cols = 1, 1
            data = data.unsqueeze(0)
        elif data.ndim == 4:
            rows, cols = 1, data.shape[0]
            data = data
        elif data.ndim == 5:
            rows, cols = data.shape[0], data.shape[1]
            data = data.flatten(0, 1)
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

        reduce_method = 't-sne'

        # 使用降维获取特征图表示
        feats_for_reduce = [data]
        reduced_feats_3d, _ = reduce_dimensionality(feats_for_reduce, dim=3, method=reduce_method, **self.reduce_params[reduce_method])  # 降维到3D用于图像显示
        data_3d = reduced_feats_3d[0]


        # [plot image]
        if image_plot:
            data = data_3d.reshape(rows, cols, *data_3d.shape[-3:])

            h, w = data.shape[-2:]
            width = cols + self.gap * (cols - 1)
            height = rows + self.gap * (rows - 1)
            fig, axes = self._subplots(rows, cols, figsize=(width, height * h / w))

            # [plot images]
            for i in range(rows):
                for j in range(cols):
                    image = data[i, j]
                    ax = axes[i, j]

                    self._remove_axes(ax)
                    # cmap = cm.get_cmap("Paired")
                    # norm = mcolors.Normalize(vmin=0, vmax=label.max().item())
                    image = tensor2numpy(image, reverse_channels=False)
                    ax.imshow(image)

            self._grid_save(fig_name, fig=fig)
            fig.clear()


        # [plot scatter plot]
        if scatter_plot:
            # Perform PCA again to reduce to 2D for scatter plot
            reduced_feats_2d, _ = reduce_dimensionality(
                feats_for_reduce, dim=2, method=reduce_method,  normalize=False,
                **self.reduce_params[reduce_method]
            )
            data_2d = reduced_feats_2d[0]
            data_3d = data_3d

            # Prepare scatter plot data
            points = data_2d.reshape(2, -1).T  # 2D coordinates for all points
            colors = data_3d.reshape(3, -1).T  # 3D color values for all points

            # Create standalone scatter plot figure
            scatter_fig, scatter_axes = self._subplots(1, 1, figsize=(10, 10))
            scatter_ax = scatter_axes[0, 0]

            # Plot scatter points
            scatter_ax.scatter(points[:, 0], points[:, 1], c=colors, s=50, alpha=1.0)

            # # Add color bar explanation
            # from mpl_toolkits.axes_grid1 import make_axes_locatable
            # divider = make_axes_locatable(scatter_ax)
            # cax = divider.append_axes("right", size="5%", pad=0.05)
            #
            # # Create color bar
            # gradient = np.linspace(0, 1, 256).reshape(1, -1)
            # gradient = np.vstack((gradient, gradient))
            # cax.imshow(gradient, aspect='auto', cmap='viridis')
            # cax.set_title('PCA 3', fontsize=8)
            # cax.set_xticks([])
            # cax.set_yticks([])

            # Save scatter plot
            scatter_fig_name = self._append_fig_name(fig_name, scatter_fig_alias)
            self._grid_save(scatter_fig_name, fig=scatter_fig)
            scatter_fig.clear()

    # noinspection PyUnresolvedReferences
    @detach_all
    @save_only_method
    @thread_parallel()
    def semantic_feature_map_joint(
            self,
            data, *joint_data,
            fig_name=None,
            reduce_method="pca",

            # image plot options
            image_plot=True, fig_alias=None,

            # scatter plot options
            scatter_plot=False, scatter_fig_alias="2d+",

            rearrange_option=None, **rearrange_kwargs
    ):
        """
        将 joint_data 在 data 的特征空间进行可视化

        使用指南：
        * 输入 data 为潜空间特征，形状为 (B, C, H, W)
        * 输入 joint_data 也为潜空间特征，形状和 data 一致，可以输入多个张量
        * fig_name：data 保存为的可视化图像的名称，最后如果带有 '+' 则会追加时间戳（比如 'image+'），如果不带有 '+' 会默认覆盖同名文件
        * fig-alias：joint_data 保存为的可视化图像的名称，数量要和 joint_data + data 的数量一致

        Args:
            data (torch.Tensor): (C, H, W), (B, C, H, W) or (B, T, C, H, W)
            joint_data (list[torch.Tensor]): (C, H, W), (B, C, H, W) or (B, T, C, H, W), other resulution data, pca at the same feature space as data
            fig_name (str): Figure name (default: None).
            image_plot (bool): Whether to plot image (default: True).
            fig_alias (str): Figure alias (default: None).
            scatter_plot (bool): Whether to plot scatter plot (default: True).
            scatter_fig_alias (str): Scatter plot figure alias (default: "2d+").
            reduce_method (str): Dimensionality reduction method (default: "pca").
            rearrange_option (str, optional): Einops-style rearrangement string.
            **rearrange_kwargs: Additional arguments for `rearrange`.
        """
        if joint_data is None:
            joint_data = []
        if isinstance(joint_data, tuple):
            joint_data = list(joint_data)

        if rearrange_option is not None:
            data = rearrange(data, rearrange_option, **rearrange_kwargs)

        for d in joint_data:
            assert data.ndim == d.ndim, f"data and joint_data must have the same number of dimensions, but got {data.ndim} and {joint_data.ndim}"
        if data.ndim == 3:
            rows, cols = 1, 1
            data = data.unsqueeze(0)
            joint_data = [d.unsqueeze(0) for d in joint_data]
        elif data.ndim == 4:
            rows, cols = 1, data.shape[0]
            data = data
            joint_data = joint_data
        elif data.ndim == 5:
            rows, cols = data.shape[0], data.shape[1]
            data = data.flatten(0, 1)
            joint_data = [d.flatten(0, 1) for d in joint_data]
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

        # Perform dimansional reduction for visualization
        feats_for_reduce = [data] + joint_data


        # [plot images]
        if image_plot:
            reduced_feats_3d, _ = reduce_dimensionality(feats_for_reduce, dim=3, method=reduce_method, **self.reduce_params[reduce_method])  # pca outputs a list of reduced tensors
            data_3d, joint_data_3d = reduced_feats_3d[0], reduced_feats_3d[1:]

            _data_3d = data_3d.reshape(cols, rows, *data_3d.shape[-3:])
            _joint_data_3d = [d.reshape(cols, rows, *d.shape[-3:]) for d in joint_data_3d]

            def plot(data, fig_name):
                h, w = data.shape[-2:]
                width = cols + self.gap * (cols - 1)
                height = rows + self.gap * (rows - 1)
                fig, axes = self._subplots(cols, rows, figsize=(width, height * h / w))

                for i in range(cols):
                    for j in range(rows):
                        image = data[j, i]
                        ax = axes[i, j]

                        self._remove_axes(ax)
                        # cmap = cm.get_cmap("Paired")
                        # norm = mcolors.Normalize(vmin=0, vmax=label.max().item())
                        image = tensor2numpy(image, reverse_channels=False)
                        ax.imshow(image)

                self._grid_save(fig_name, fig=fig)
                fig.clear()

            datas = [_data_3d] + _joint_data_3d
            if not isinstance(fig_alias, (list, tuple)):
                if fig_alias is None:
                    fig_alias = [f"{i}" for i in range(len(datas))]
                else:
                    fig_alias = [f"{fig_alias}_{i}" for i in range(len(datas))]
            else:
                assert len(fig_alias) == len(datas), f"fig_alias must have the same length as datas, but got {len(fig_alias)} and {len(datas)}"
            for i, d in enumerate(datas):
                plot(d, self._append_fig_name(fig_name, fig_alias[i]))


        # [plot scatter plot]
        if scatter_plot:
            scatter_fig_name = self._append_fig_name(fig_name, scatter_fig_alias)
            self._reduce_and_visualize(
                feats_for_reduce, fig_name=scatter_fig_name, reduce_method=reduce_method,
                scatter_size=1
            )
            # # Perform PCA again to reduce to 2D for scatter plot
            # reduced_feats_2d, _ = reduce_dimensionality(
            #     feats_for_reduce, dim=2, method=reduce_method, normalize=False,
            #     **self.reduce_params[reduce_method]
            # )  # pca outputs a list of reduced tensors
            # data_2d, joint_data_2d = reduced_feats_2d[0], reduced_feats_2d[1:]
            #
            # # Prepare scatter plot data
            # data_2d = data_2d.reshape(2, -1).T
            # joint_data_2d = [d.reshape(2, -1).T for d in joint_data_2d]
            # # data_3d = data_3d.reshape(3, -1).T
            # # joint_data_3d = [d.reshape(3, -1).T for d in joint_data_3d]
            #
            # # Create standalone scatter plot figure
            # scatter_fig, scatter_ax = plt.subplots(figsize=(10, 10))
            # scatter_ax: plt.Axes = scatter_ax
            #
            # # Plot scatter points
            # markers = ['o', 's', '^', 'v', '<', '>', '+', 'x', 'D', 'p']
            # # for i in range(len(joint_data_2d)):
            # #     scatter_ax.scatter(joint_data_2d[i][:, 0], joint_data_2d[i][:, 1], c=joint_data_3d[i], s=50, alpha=1.0, marker=markers[i % len(markers)])
            # # scatter_ax.scatter(data_2d[:, 0], data_2d[:, 1], c=data_3d, s=100, alpha=1.0, marker='*')
            # colors = ['blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            # for i in range(len(joint_data_2d)):
            #     scatter_ax.scatter(joint_data_2d[i][:, 0], joint_data_2d[i][:, 1], color=colors[i % len(colors)], s=10, alpha=0.1, marker=markers[i % len(markers)])
            # scatter_ax.scatter(data_2d[:, 0], data_2d[:, 1], color='red', s=10, alpha=0.1, marker='*')
            #
            # # Get current axis limits to determine range
            # xlim = scatter_ax.get_xlim()
            # ylim = scatter_ax.get_ylim()
            #
            # # Calculate the maximum range for symmetric axes
            # x_range = max(xlim[1], -xlim[0])
            # y_range = max(ylim[1], -ylim[0])
            # max_range = max(x_range, y_range)
            #
            # # Set symmetric limits around origin (0,0) - place (0,0) at image center
            # scatter_ax.set_xlim(-max_range, max_range)
            # scatter_ax.set_ylim(-max_range, max_range)
            #
            # # Draw coordinate axes through the origin (0,0)
            # scatter_ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
            # scatter_ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8, alpha=0.7)
            #
            # # Set equal aspect ratio to make it square
            # scatter_ax.set_aspect('equal')
            #
            # # Save scatter plot
            # scatter_fig_name = self._append_fig_name(fig_name, scatter_fig_alias)
            # self._grid_save(scatter_fig_name)
            # plt.close(scatter_fig)

    # noinspection PyUnresolvedReferences
    @detach_all
    @save_only_method
    @thread_parallel()
    def threshold_feature_map_joint(
            self,
            data, *joint_data,
            fig_name=None, fig_alias=None,
            threshold=0.5, threshold_dir="positive",
            reduce_method="pca",
            rearrange_option=None, **rearrange_kwargs
    ):
        """
        将 joint_data 在 data 的特征空间进行可视化，并剔除背景

        使用指南：
        * 输入 data 为潜空间特征，形状为 (B, C, H, W)
        * 输入 joint_data 也为潜空间特征，形状和 data 一致，可以输入多个张量
        * fig_name：data 保存为的可视化图像的名称，最后如果带有 '+' 则会追加时间戳（比如 'image+'），如果不带有 '+' 会默认覆盖同名文件
        * fig-alias：joint_data 保存为的可视化图像的名称，数量要和 joint_data + data 的数量一致

        Args:
            data (torch.Tensor): (C, H, W), (B, C, H, W) or (B, T, C, H, W)
            joint_data (list[torch.Tensor]): (C, H, W), (B, C, H, W) or (B, T, C, H, W), other resulution data, pca at the same feature space as data
            fig_name (str): Figure name (default: None).
            rearrange_option (str, optional): Einops-style rearrangement string.
            **rearrange_kwargs: Additional arguments for `rearrange`.
        """
        if joint_data is None:
            joint_data = []
        if isinstance(joint_data, tuple):
            joint_data = list(joint_data)

        if rearrange_option is not None:
            data = rearrange(data, rearrange_option, **rearrange_kwargs)

        for d in joint_data:
            assert data.ndim == d.ndim, f"data and joint_data must have the same number of dimensions, but got {data.ndim} and {joint_data.ndim}"
        if data.ndim == 3:
            rows, cols = 1, 1
            data = data.unsqueeze(0)
            joint_data = [d.unsqueeze(0) for d in joint_data]
        elif data.ndim == 4:
            rows, cols = 1, data.shape[0]
            data = data
            joint_data = joint_data
        elif data.ndim == 5:
            rows, cols = data.shape[0], data.shape[1]
            data = data.flatten(0, 1)
            joint_data = [d.flatten(0, 1) for d in joint_data]
        else:
            raise ValueError(f"Unsupported data shape: {data.shape}")

        # Perform dimansional reduction for visualization
        feats_for_reduce = [data] + joint_data
        reduced_feats_1d, _ = reduce_dimensionality(
            feats_for_reduce, dim=1, method=reduce_method, normalize=True, **self.reduce_params[reduce_method]
        )  # pca outputs a list of reduced tensors

        bg_feats_for_reduce = []
        for data in bg_feats_for_reduce:
            data = data.clone()
            if threshold_dir:
                data[data < threshold] = 0
            else:
                data[data > threshold] = 0
            bg_feats_for_reduce.append(data)

        reduced_feats_3d, _ = reduce_dimensionality(bg_feats_for_reduce, dim=3, method=reduce_method, **self.reduce_params[reduce_method])
        data_3d, joint_data_3d = reduced_feats_3d[0], reduced_feats_3d[1:]

        _data_3d = data_3d.reshape(cols, rows, *data_3d.shape[-3:])
        _joint_data_3d = [d.reshape(cols, rows, *d.shape[-3:]) for d in joint_data_3d]

        # [plot images]
        def plot(data, fig_name):
            h, w = data.shape[-2:]
            width = cols + self.gap * (cols - 1)
            height = rows + self.gap * (rows - 1)
            fig, axes = self._subplots(cols, rows, figsize=(width, height * h / w))

            for i in range(cols):
                for j in range(rows):
                    image = data[j, i]
                    ax = axes[i, j]

                    self._remove_axes(ax)
                    # cmap = cm.get_cmap("Paired")
                    # norm = mcolors.Normalize(vmin=0, vmax=label.max().item())
                    image = tensor2numpy(image, reverse_channels=False)
                    ax.imshow(image)

            self._grid_save(fig_name, fig=fig)
            fig.clear()

        datas = [_data_3d] + _joint_data_3d
        if not isinstance(fig_alias, (list, tuple)):
            if fig_alias is None:
                fig_alias = [f"{i}" for i in range(len(datas))]
            else:
                fig_alias = [f"{fig_alias}_{i}" for i in range(len(datas))]
        else:
            assert len(fig_alias) == len(datas), f"fig_alias must have the same length as datas, but got {len(fig_alias)} and {len(datas)}"
        for i, d in enumerate(datas):
            plot(d, self._append_fig_name(fig_name, fig_alias[i]))

    # noinspection PyUnresolvedReferences
    @detach_all
    @save_only_method
    @thread_parallel()
    def attention_map(
            self, x, attention, fig_name=None, attention_cmap='jet',
            alpha_gradiant=True,
            rearrange_option=None, **rearrange_kwargs
    ):
        """
        在 x 上叠加 attention 图（注意力图），其中绘制出来的 attention 图是半透明的：(B, C, H, W)+(B, 1, H, W) -> (B, C, H, W)
            其中：要满足 x 是 RGB 图像或者灰度图
        Args:
            x (torch.Tensor): (C, H, W) or (B, C, H, W)
            attention (torch.Tensor): (1, H, W) or (B, 1, H, W)
            fig_name (str): Figure name (default: None).
            attention_cmap (str): Matplotlib colormap for attention map (default: 'jet').
            rearrange_option (str, optional): Einops-style rearrangement string.
            alpha_gradiant (bool): Whether to use alpha gradiant for attention map (default: True).
            **rearrange_kwargs: Additional arguments for `rearrange`.
        """
        h, w = x.shape[-2:]

        if rearrange_option is not None:
            x = rearrange(x, rearrange_option, **rearrange_kwargs)
            attention = rearrange(attention, rearrange_option, **rearrange_kwargs)

        if x.ndim == 3:
            x = x.unsqueeze(0)
        if attention.ndim == 3:
            attention = attention.unsqueeze(0)

        # [plot images]
        num = x.shape[0]
        width = num + self.gap * (num - 1)
        height = 1
        fig, axes = self._subplots(num, 1, figsize=(width, height * h / w))

        for i in range(num):
            image = x[i]
            mask = attention[i]
            ax = axes[i, 0]

            self._remove_axes(ax)

            # 将注意力图叠加到图像上
            image_h, image_w = image.shape[-2:]
            image = tensor2numpy(image, reverse_channels=False)
            ax.imshow(image, alpha=1)

            if alpha_gradiant:
                mask_gray = tensor2numpy((mask / mask.max()) * 0.2 + 0.3)
            else:
                mask_gray = np.ones_like(mask.permute(1, 2, 0)) * 255 * 0.3
            mask_cmap = self._grayscale_to_heatmap(mask, attention_cmap, normalize=True)
            mask = np.concatenate([mask_cmap, mask_gray], axis=-1)
            mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
            mask = (mask / mask.max() * 255).round().astype(np.uint8)
            ax.imshow(mask)

        self._grid_save(fig_name, fig=fig)
        fig.clear()

    # noinspection PyUnresolvedReferences
    @detach_all
    @save_only_method
    @thread_parallel()
    def attention_map_selection(
            self, x, *attentions,
            windows_names=None, save_names=None,
            fig_name=None, attention_cmap=cv2.COLORMAP_JET
    ):
        """
        在 x 上叠加 attention 图阵（注意力图阵），其中绘制出来的 attention 图是半透明的：(C, H, W)+(1, H, W) -> (C, H, W)
            其中：要满足 x 是 RGB 图像或者灰度图
        Args:
            x (torch.Tensor): (C, H, W)
            attentions (torch.Tensor): (1, HW, HW)
            fig_name (str): Figure name (default: None).
            attention_cmap: Matplotlib colormap for attention map (default: 'jet').
        """
        h0, w0 = x.shape[-2:]
        n_scale = math.sqrt(h0 * w0 / attentions[0].shape[-1])
        h, w = round(h0 / n_scale), round(w0 / n_scale)
        assert (h0 * w0) % (h * w) == 0, f"h0 * w0 must be divisible by h * w, but got {h0 * w0} and {h * w}"

        if isinstance(attentions, tuple):
            attentions = list(attentions)

        if x.ndim == 3:
            x = x.unsqueeze(0)
        for i in range(len(attentions)):
            if attentions[i].ndim == 3:
                attentions[i] = attentions[i].unsqueeze(0)

        if x.ndim == 4:
            assert x.shape[0] == 1, f"x must be (1, C, H, W) or (C, H, W), but got {x.shape}"
        for i in range(len(attentions)):
            attention = attentions[i]
            if attention.ndim == 4:
                assert attention.shape[0] == 1, f"attention must be (1, 1, H, W) or (1, H, W), but got {attention.shape}"

        for i in range(len(attentions)):
            attention = attentions[i]

            attention = rearrange(attention, 'b c (h1 w1) (h2 w2) -> b c h1 w1 h2 w2', h1=h, w1=w, h2=h, w2=w)
            if torch.max(attention) > 1 or torch.min(attention) < 0:
                attention = (attention - torch.min(attention)) / (torch.max(attention) - torch.min(attention))
                try:
                    from .log import get_root_logger
                    logger = get_root_logger()
                    logger.warning(f"Attention map should be in range [0, 1], but got {torch.min(attention)} and {torch.max(attention)}, normalized to [0, 1]")
                except:
                    print(f"Attention map should be in range [0, 1], but got {torch.min(attention)} and {torch.max(attention)}, normalized to [0, 1]")

            attentions[i] = attention

        if windows_names is None:
            windows_names = [f"{i}" for i in range(len(attentions))]
        # [cv display images]
        def on_mouse_click(event, x, y, flags, param):
            if event in [cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE] and (flags & cv2.EVENT_FLAG_LBUTTON):
                window_name = param
                image_ref: np.ndarray = displayed_images[window_name]
                if image_ref is not None and y < image_ref.shape[0] and x < image_ref.shape[1]:
                    _x, _y = round(x / n_scale), round(y / n_scale)

                    attn_maps = []
                    for i in range(len(attentions)):
                        attention = attentions[i]

                        # get attention map at x, y
                        attn_map = attention[:, :, _y, _x, :, :]   # (C, H, W, H, W) -> (C, H, W)
                        attn_map = attn_map / torch.max(attn_map)
                        attn_map = F.interpolate(attn_map, size=(h0, w0), mode='area')

                        # attention overlay
                        attn_map = tensor2numpy(attn_map)
                        attn_map = cv2.applyColorMap(attn_map, attention_cmap)
                        image_out = cv2.addWeighted(image_ref, 0.5, attn_map, 0.5, 0)

                        displayed_images[windows_names[i]] = image_out
                        attn_maps.append(attn_map)
                    save_images[f"attn"] = attn_maps

        displayed_images = {}
        save_images = {}
        window_name_ref = 'reference'
        displayed_images[window_name_ref] = tensor2numpy(x)
        any_gt_saved = False

        cv2.imshow(window_name_ref, displayed_images[window_name_ref])
        cv2.setMouseCallback(window_name_ref, on_mouse_click, param=window_name_ref)

        for window_name_out in windows_names:
            displayed_images[window_name_out] = np.zeros_like(displayed_images[window_name_ref])
            cv2.imshow(window_name_out, displayed_images[window_name_out])
            cv2.setMouseCallback(window_name_out, on_mouse_click, param=window_name_ref)

        while True:
            cv2.imshow(window_name_ref, displayed_images[window_name_ref])

            for window_name_out in windows_names:
                cv2.imshow(window_name_out, displayed_images[window_name_out])

            key = cv2.waitKey(30)
            if key == 27:  # ESC key
                break
            elif key == ord('s') or key == ord('S'):  # 's' or 'S' key
                if not any_gt_saved:
                    save_name = self._append_fig_name(fig_name, "ref")
                    self._cv_save(displayed_images[window_name_ref], save_name)
                for window_name_out in windows_names:
                    if save_names is not None:
                        save_name = self._append_fig_name(fig_name, save_names[windows_names.index(window_name_out)])
                    else:
                        save_name = self._append_fig_name(fig_name, "vis")
                    self._cv_save(displayed_images[window_name_out], save_name)
                if "attn" in save_images:
                    for i in range(len(save_images["attn"])):
                        attn = save_images["attn"][i]
                        window_name_out = windows_names[i]
                        if save_names is not None:
                            save_name = self._append_fig_name(fig_name, save_names[windows_names.index(window_name_out)])
                        else:
                            save_name = self._append_fig_name(fig_name, window_name_out)
                        self._cv_save(attn, save_name)
                print(f"Attention map saved")
                any_gt_saved = True
            elif key in (255, 127, 8):  # backspace, delete, and left arrow key for exiting
                return True
        return False

    # noinspection PyUnresolvedReferences
    @detach_all
    @save_only_method
    @thread_parallel()
    def similarity_map(
            self, x, y, fig_name=None, cmap='viridis',
            similarity_type='cosine', map_type='figure',
            similarity_post_process_func=None,
            rearrange_option=None, **rearrange_kwargs
    ):
        """
        计算 x 和 y 的相似度并可视化：
         - 如果使用 figure 模式，则绘制对角巷相似度图（传统的相似度图）
         - 如果使用 per_image 模式，逐像素空间相似度图（对每个特征计算全局相似度，得到类似注意力的效果）
        Args:
            x (torch.Tensor): (B, C, H, W) or (B, 1, C, H, W)
            y (torch.Tensor): (B, C, H, W) or (B, T, C, H, W)
            fig_name (str): Figure name (default: None).
            cmap (str): Matplotlib colormap for similarity map (default: 'jet').
            rearrange_option (str, optional): Einops-style rearrangement string.
            similarity_type (str): Similarity type, can be 'cosine' or 'L2' (default: 'cosine').
            map_type (str): Map type, can be 'figure' or 'per_image' (default: 'figure').
            similarity_post_process_func (callable, optional): Post-process function for similarity map.
            **rearrange_kwargs: Additional arguments for `rearrange`.
        """
        b = x.shape[0]
        h, w = x.shape[-2:]

        # Handle time dimension in y (B, T, C, H, W)
        has_time_dim = y.dim() == 5 and y.size(1) >= 1

        if map_type == 'per_image':
            assert b == 1, "Only support batch size 1 when map_type is 'per_image'"
            if has_time_dim:
                assert y.size(1) == 1, "For per_image mode with time dimension, T must be 1"

        if y.dim() == 4:
            x = rearrange(x, "b c h w -> b (h w) c")
            y = rearrange(y, "b c h w -> b (h w) c")
        elif y.dim() == 5:
            x = rearrange(x, "b t c h w -> b t (h w) c")
            y = rearrange(y, "b t c h w -> b t (h w) c")

        # Handle time dimension in similarity calculation
        if has_time_dim:
            # For x (B,1,C,H,W) and y (B,T,C,H,W), compute similarity between each time step
            similarity = []
            for t in range(y.shape[1]): # 串行是为了保证显存不爆炸
                sim = _get_sim(x.squeeze(1), y[:, t], similarity_type)  # (B, N, N)
                similarity.append(sim)
            similarity = torch.stack(similarity, dim=1)  # (B, T, N, N)
        else:
            similarity = _get_sim(x, y, similarity_type)  # (B, N, N)

        if rearrange_option is not None:
            similarity = rearrange(similarity, rearrange_option, **rearrange_kwargs)

        if map_type == 'figure':
            if has_time_dim:
                # For time dimension, create a grid of subplots for each time step
                t = similarity.shape[1]
                rows = b
                cols = t
                width = cols + self.gap * (cols - 1)
                height = rows + self.gap * (rows - 1)
                fig, axes = self._subplots(cols, rows, figsize=(width, height))

                for i in range(cols):
                    for j in range(rows):
                        ax = axes[i, j]

                        self._remove_axes(ax)

                        similarity_map = similarity[i, j]
                        if similarity_post_process_func is not None:
                            similarity_map = similarity_post_process_func(similarity_map)
                        ax.imshow(similarity_map, cmap=cmap)
            else:
                cols = b + self.gap * (b - 1)
                fig, axes = self._subplots(1, b, figsize=(cols, 1))
                for i in range(b):
                    ax = axes[0, i]

                    self._remove_axes(ax)

                    similarity_map = similarity[i]
                    if similarity_post_process_func is not None:
                        similarity_map = similarity_post_process_func(similarity_map)
                    ax.imshow(similarity_map, cmap=cmap)

        elif map_type == 'per_image':
            # (1, N, N) -> (H, W, H, W)
            similarity = rearrange(similarity, "b (h w) (h2 w2) -> b h w h2 w2", h=h, w=w, h2=h, w2=w)[0]
            cols, rows = w, h
            width = cols + self.gap * (cols - 1)
            height = rows + self.gap * (rows - 1)
            fig, axes = self._subplots(cols, rows, figsize=(width, height * h / w))

            # [plot images]
            for i in range(cols):
                for j in range(rows):
                    similarity_map = similarity[j, i]
                    if similarity_post_process_func is not None:
                        similarity_map = similarity_post_process_func(similarity_map)
                    ax = axes[i, j]

                    self._remove_axes(ax)
                    ax.imshow(similarity_map, cmap=cmap)

        else:
            raise ValueError(f"Unsupported map_type: {map_type}")

        self._grid_save(fig_name, fig=fig)
        fig.clear()


    #region ==[Temporal]==
    @detach_all
    @save_only_method
    @thread_parallel()
    def temporal_difference_map(
            self, x, fig_name=None, cmap='jet',
            rearrange_option=None, **rearrange_kwargs
    ):
        """
        计算 x 第 t 帧和第 t-1 帧的差异图（t in [1, T]）
        Args:
            x (torch.Tensor): (B, T, C, H, W)
            cmap (str): Matplotlib colormap for difference map (default: 'jet').
            rearrange_option (str, optional): Einops-style rearrangement string.
            **rearrange_kwargs: Additional arguments for `rearrange`.
        """
        pass


    @detach_all
    @save_only_method
    @thread_parallel()
    def mabd_vector_plot(
            self, *videos, fig_name=None,
            labels=None
    ):
        """
        Args:
            videos (torch.Tensor): (B, T, C, H, W)
        """
        if labels is not None:
            assert len(videos) == len(labels), f"Number of videos and labels must be the same. But got {len(videos)} videos and {len(labels)} labels."

        from basic.metrics.mabd import get_mabd_vector
        mabd_vectors = [get_mabd_vector(x) for x in videos]

        # plot mabd vectors in one figure
        fig, axes = self._subplots(1, 1, figsize=(6, 6))
        ax = axes[0, 0]
        for i, mabd_vector in enumerate(mabd_vectors):
            label = labels[i] if labels is not None else f"video {i}"
            ax.plot(mabd_vector.squeeze(0), label=label)
        ax.set_title("Flickering Effect Measurement")
        ax.set_xlabel("Frame number")
        ax.set_ylabel("MABD between each frame")
        ax.legend()

        self._grid_save(fig_name, settings='graph', fig=fig)
        fig.clear()
    #endregion


    @detach_all
    @save_only_method
    @thread_parallel()
    def grid_wrapper(
            self,
            data_list,
            plot_function,
            fig_name=None,
            figsize=(6, 6),
            nrows=1, ncols=3,
            outter_grid_kwargs=None,
            **kwargs
    ):
        """

        Args:
            plot_function (callable): Plot function to be called for each data item.
            data_list (list of torch.Tensor of torch.Tensor): List of data items to be plotted.
            figsize (tuple, optional): Figure size. Defaults to (6, 6).
            nrows (int, optional): Number of rows in the grid. Defaults to 1.
            ncols (int, optional): Number of columns in the grid. Defaults to 3.
            outter_grid_kwargs (dict, optional): Keyword arguments for the outer grid layout. Defaults to None.

        e.g.
            nrows, ncols = 2, 3
            data_list = [torch.randn(2, 3, 16, 16) for _ in range(nrows * ncols)]
        """
        from unittest import mock
        from basic.utils.general import get_original_callable
        import matplotlib.gridspec as gridspec

        if outter_grid_kwargs is None:
            outter_grid_kwargs = dict(wspace=0.05, hspace=0.05 * ncols / nrows)
        sub_figsize_list = []
        plot_function = get_original_callable(plot_function)

        # 创建方格图大图
        fig = Figure(figsize=figsize, facecolor='none')
        FigureCanvasAgg(fig)
        grid = gridspec.GridSpec(nrows, ncols, figure=fig, **outter_grid_kwargs)

        # 将子图的部分功能函数禁用和替代
        def plt_subplots(nrows=1, ncols=1, *args, **kwargs):
            sub_figsize = kwargs.get("figsize", None)
            if sub_figsize is not None:
                sub_figsize_list.append(sub_figsize)

            i = _counter
            inner_grid = gridspec.GridSpecFromSubplotSpec(
                nrows, ncols,
                subplot_spec=grid[i],
            )

            axes = np.array([[fig.add_subplot(inner_grid[r, c]) for c in range(ncols)] for r in range(nrows)])
            axes = axes.squeeze()  # squeeze to act as subplots
            return fig, axes

        def plt_close(*args, **kwargs):
            pass

        def plt_save(*args, **kwargs):
            pass

        # 绘制子图
        _counter = 0
        with mock.patch("matplotlib.pyplot.subplots", plt_subplots), \
                mock.patch("matplotlib.pyplot.close", plt_close), \
                mock.patch("matplotlib.pyplot.savefig", plt_save):
            for i, data in enumerate(data_list):
                _counter = i
                plot_function(data, fig_name=None, **kwargs)

        # 根据子图自适应调整图像大小
        if sub_figsize_list:
            sub_w = max(fs[0] for fs in sub_figsize_list)
            sub_h = max(fs[1] for fs in sub_figsize_list)

            total_w = ncols * sub_w * (1 - (outter_grid_kwargs['wspace'] / ncols)) / (1 - outter_grid_kwargs['wspace'])
            total_h = nrows * sub_h * (1 - (outter_grid_kwargs['hspace'] / nrows)) / (1 - outter_grid_kwargs['hspace'])

            fig.set_size_inches(total_w, total_h, forward=True)

        self._grid_save(fig_name, fig=fig)
        fig.clear()


    #region ==[VGG feature map]==
    def _try_initialize_vgg_model(self, device):
        """Initialize VGG model if not already done."""
        if not hasattr(self, '_vgg_model'):
            self._vgg_model = None

        if self._vgg_model is None:
            self._vgg_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()

        self._vgg_model.to(device)

    def _extract_vgg_features(self, images: List[torch.Tensor], feature_dim_mode: str = 'mean'):
        """Extract VGG features from images.

        # example:
        from basic.utils.console.logplot import AccumulatedPlotter
        plotter = AccumulatedPlotter("./.plotlogs")
        plotter.vgg_feature_map("vgg/pred+", [preds[:, i] for i in range(preds.shape[1])])
        """
        self._try_initialize_vgg_model(images[0].device)

        all_features = []
        for img in images:
            if img.dim() == 3:
                img = img.unsqueeze(0)

            with torch.no_grad():
                features = self._vgg_model(img) # [1, 512, 8, 8] for input size 256x256

            if feature_dim_mode == 'flatten':
                features_reduced = features.view(features.size(0), -1)
            elif feature_dim_mode == 'mean':
                features_reduced = features.mean(dim=(2, 3))
            elif feature_dim_mode == 'max':
                features_reduced = features.max(dim=(2, 3))[0]
            elif feature_dim_mode == 'min':
                features_reduced = features.min(dim=(2, 3))[0]
            elif feature_dim_mode == 'median':
                features_reduced = features.median(dim=(2, 3))[0]
            else:
                raise ValueError(f"Unsupported feature_dim_mode: {feature_dim_mode}")
            all_features.append(features_reduced)

        return all_features

    @detach_all
    @save_only_method
    @thread_parallel()
    def vgg_feature_map(
            self, fig_name: str,
            images: List[torch.Tensor],
            reduce_method: str = 'pca',
    ):
        """
        Accumulate VGG feature maps and perform dimensionality reduction visualization.

        # accumulated example:
        from basic.utils.console.logplot import AccumulatedPlotter
        plotter = AccumulatedPlotter("./.plotlogs")
        plotter.vgg_feature_map("vgg/pred+", [preds[:, i] for i in range(preds.shape[1])])

        Args:
            fig_name (str): Name of the figure
            images (list[torch.Tensor]): different images (B, C, H, W)
            reduce_method (str): Dimensionality reduction method ('pca', 'tsne', 'umap')
        """
        projected_features = self._extract_vgg_features(images)
        self._reduce_and_visualize(projected_features, fig_name=fig_name, reduce_method=reduce_method)

    @detach_all
    @save_only_method
    @thread_parallel()
    def vgg_feature_temporal_map(
            self, fig_name: str,
            videos: List[torch.Tensor], *joint_videos,
            reduce_method: str = 'pca',
    ):
        """
        Accumulate VGG feature maps and perform dimensionality reduction visualization.

        # accumulated example:
        from basic.utils.console.logplot import AccumulatedPlotter
        plotter = AccumulatedPlotter("./.plotlogs")
        plotter.accumulate("pred", preds)
        def function(x):
            plotter.vgg_feature_temporal_map("vgg/pred/+", x, 'pca')
        plotter.accumulated_operate( "pred", function, disposable=False)

        Args:
            fig_name (str): Name of the figure
            videos (list[torch.Tensor]): different videos (B, T, C, H, W)
            reduce_method (str): Dimensionality reduction method ('pca', 'tsne', 'umap')
        """
        videos = [rearrange(v, "b t c h w -> (b t) c h w") for v in videos]
        projected_features = self._extract_vgg_features(videos)
        # m*(N, dim) -> (m, N, dim) -> (N, m, dim); -> N*(m, dim)
        projected_features = torch.stack(projected_features, dim=0).transpose(0, 1)
        projected_features = [f for f in projected_features]
        self._reduce_and_visualize(projected_features, fig_name=fig_name, reduce_method=reduce_method, gradient=True)

    @detach_all
    @save_only_method
    @thread_parallel()
    def vgg_feature_temporal_map_joint(
            self, fig_name: str,
            videos: List[torch.Tensor], *joint_videos,
            reduce_method: str = 'pca',
    ):
        """
        Accumulate VGG feature maps and perform dimensionality reduction visualization.

        # accumulated example:
        from basic.utils.console.logplot import get_root_plotter, AccumulatedPlotter
        plotter = get_root_plotter("./.plotlogs", plotter_class=AccumulatedPlotter.__name__)
        plotter.accumulate("pred", preds)
        plotter.accumulate("gt", gts)
        def function(*x):
            plotter.vgg_feature_temporal_map_joint("vgg/pred-gt/+", *x, reduce_method='pca')
        plotter.accumulated_operate_joint(["pred", "gt"], function, disposable=False)

        Args:
            fig_name (str): Name of the figure
            videos (list[torch.Tensor]): different videos (B, T, C, H, W)
            reduce_method (str): Dimensionality reduction method ('pca', 'tsne', 'umap')
        """
        def accumulate_video(videos):
            videos = [rearrange(v, "b t c h w -> (b t) c h w") for v in videos]
            projected_features = self._extract_vgg_features(videos)
            # m*(N, dim) -> (m, N, dim) -> (N, m, dim); -> N*(m, dim)
            projected_features = torch.stack(projected_features, dim=0).transpose(0, 1)
            projected_features = [f for f in projected_features]
            return projected_features

        # [accumulate videos]
        projected_features = accumulate_video(videos)

        # [accumulate joint videos]
        projected_joint_features = []
        for joint_video in joint_videos:
            joint_features = accumulate_video(joint_video)
            projected_joint_features.append(joint_features)

        self._reduce_and_visualize(
            projected_features, *projected_joint_features,
            fig_name=fig_name, reduce_method=reduce_method, gradient=True
        )
    #endregion


class AccumulatedPlotter(Plotter):
    def __init__(self, root, dpi=256, gap=0.02):
        super().__init__(root, dpi, gap)

        self._accumulated_images = {}

    @detach_all
    @save_only_method
    @thread_parallel()
    def accumulate(self, key, image):
        """Accumulate images for a given figure name."""
        if key not in self._accumulated_images:
            self._accumulated_images[key] = []
        self._accumulated_images[key].append(image)

    @detach_all
    @save_only_method
    @thread_parallel()
    def accumulated_operate(
            self, key: str,
            function,
            *function_args,
            inputs: Optional[torch.Tensor] = None,
            parameter_type: str='tensor_list',
            disposable: bool=True,
            **function_kwargs
    ):
        if inputs is not None:
            self.accumulate(key, inputs)

        if key in self._accumulated_images:
            accumulated_imgs = self._accumulated_images[key]

            if parameter_type == 'tensor_list':
                function(accumulated_imgs, *function_args, **function_kwargs)  # tensor list
            elif parameter_type == 'tensors':
                function(*accumulated_imgs, *function_args, **function_kwargs) # tensors
            elif parameter_type == 'cat_tensor':
                cat_features = torch.cat(accumulated_imgs, dim=0)
                function(cat_features, *function_args, **function_kwargs)      # cat tensor
            else:
                raise ValueError(f"Unsupported parameter_type: {parameter_type}")

            if disposable:
                self._accumulated_images[key] = []
        else:
            print(f"No accumulated images found for {key}")

    @detach_all
    @save_only_method
    @thread_parallel()
    def accumulated_operate_joint(
            self, keys: List[str],
            function,
            disposable: bool=True,
    ):
        accumulated_imgs_joint = []
        for key in keys:
            if key in self._accumulated_images:
                accumulated_imgs_joint.append(self._accumulated_images[key])

                if disposable:
                    self._accumulated_images[key] = []
            else:
                print(f"No accumulated images found for {key} to joint operate.")

        function(*accumulated_imgs_joint)

    def get_accumulated_count(self, key: str) -> int:
        if key in self._accumulated_images:
            return len(self._accumulated_images[key])
        return 0

    def clear_accumulated(self, key: Optional[str] = None):
        if key is None:
            self._accumulated_images.clear()
        elif key in self._accumulated_images:
            del self._accumulated_images[key]

    def list_accumulated_figures(self) -> List[str]:
        return list(self._accumulated_images.keys())
#endregion


import torch
import torch.nn.functional as F
import math


def _get_sim(mk, qk, type='L2'):
    """
    Args:
        mk (torch.Tensor): (B, N, d), memory keys
        qk (torch.Tensor): (B, n, d), query keys

    Returns:
        torch.Tensor: (B, n, N)
    """
    B, n, d = qk.shape
    if type == 'cosine':
        keys_norm = F.normalize(mk, dim=-1)     # (B, N, d)
        query_norm = F.normalize(qk, dim=-1)    # (B, n, d)
        similarity = torch.matmul(query_norm, keys_norm.transpose(-1, -2))  # (B, n, d) @ (B, N, d)^T -> (B, n, N)
    elif type == 'dot_product':
        similarity = torch.matmul(mk, qk.transpose(-1, -2))  # (B, n, d) @ (B, N, d)^T -> (B, n, N)
    elif type == 'L2':
        a = mk.pow(2).sum(dim=2).unsqueeze(1)   # (B, N, d) -> (B, 1, N)
        b = 2 * (qk @ mk.transpose(-1, -2))     # (B, n, d) @ (B, N, d)^T -> (B, n, N)
        similarity = (-a + b) / math.sqrt(d)    # (B, n, N)
    else:
        raise ValueError(f"Unsupported similarity type: {type}")

    return similarity


#region ==[Dim. Reduction]==
import umap                         # pip install umap-learn
from sklearn.manifold import TSNE   # pip install scikit-learn
"""
Modified from JAFAR((https://github.com/PaulCouairon/JAFAR/tree/main/utils/visualization)
"""


class TorchPCA(object):

    def __init__(self, n_components, center=False, niter=4):
        self.n_components = n_components
        self.center = center
        self.niter = niter

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=self.center, niter=self.niter)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected


class TorchUMAP(object):
    def __init__(self, n_components, **kwargs):
        self.n_components = n_components
        self.umap_kwargs = kwargs

    def fit(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        self.umap = umap.UMAP(n_components=self.n_components, **self.umap_kwargs)
        self.umap.fit(X)
        return self

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        return torch.from_numpy(self.umap.transform(X))


class TorchTSNE(object):
    def __init__(self, n_components, **kwargs):
        self.n_components = n_components
        self.tsne_kwargs = kwargs

    def fit(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        n_samples = X.shape[0]
        perplexity = min(max(5, min(50, int(n_samples ** 0.5))), n_samples - 1)   # p < N - 1 or 5 <= p <= 50
        # print(f"shape of X: {X.shape}, perplexity: {perplexity}")
        self.tsne = TSNE(n_components=self.n_components, perplexity=perplexity, **self.tsne_kwargs)
        # t-SNE doesn't have a separate transform, so we store the fitted model
        return self

    def transform(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        # Note: t-SNE doesn't have a proper transform, so we fit_transform each time
        # This isn't ideal but matches t-SNE's typical usage pattern
        # print(f"shape of input X: {X.shape}")
        return torch.from_numpy(self.tsne.fit_transform(X))


def reduce_dimensionality(
        image_feats_list: list, dim=3, fit_reducer=None,
        method='pca', max_samples=None, normalize=True,
        **kwargs
):
    """
    Generic dimensionality reduction function supporting PCA, UMAP, and t-SNE

    Args:
        image_feats_list: List of feature tensors (C,), (B, C), (C, H, W), (B, C, H, W), (B, T, C, H, W)
        dim: Output dimension
        fit_reducer: Pre-fitted reducer object
        method: 'pca', 'umap', or 'tsne'
        max_samples: Maximum samples to use for fitting
        kwargs: Additional arguments for the reducer
    Returns:
        reduced_feats: List of reduced feature tensors (B, dim, H, W)
        fit_reducer: Fitted reducer object
    """
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_reducer is None:
            tensor = F.interpolate(tensor, target_size, mode="area")

        # -> (N, C)
        if tensor.dim() == 1:   # (C,)
            return tensor.unsqueeze(0)
        elif tensor.dim() == 2: # (B, C)
            return tensor
        elif tensor.dim() == 3: # (C, H, W)
            return rearrange(tensor, "c h w -> (h w) c")
        elif tensor.dim() == 4: # (B, C, H, W)
            return rearrange(tensor, "b c h w -> (b h w) c")
        elif tensor.dim() == 5: # (B, T, C, H, W)
            return rearrange(tensor, "b t c h w -> (b t h w) c")
        else:
            raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")

    def unflatten(tensor, ref):
        if ref.dim() == 1:   # (C,)
            return tensor.reshape(-1)
        elif ref.dim() == 2: # (B, C)
            return tensor
        elif ref.dim() == 3: # (C, H, W)
            C, H, W = ref.shape
            return rearrange(tensor, "(h w) c -> c h w", h=H, w=W)
        elif ref.dim() == 4: # (B, C, H, W)
            B, C, H, W = ref.shape
            return rearrange(tensor, "(b h w) c -> b c h w", b=B, h=H, w=W)
        elif ref.dim() == 5: # (B, T, C, H, W)
            B, T, C, H, W = ref.shape
            return rearrange(tensor, "(b t h w) c -> b t c h w", b=B, t=T, h=H, w=W)
        else:
            raise ValueError(f"Unsupported tensor dimension: {ref.dim()}")

    if len(image_feats_list) > 1 and fit_reducer is None and image_feats_list[0].dim() >= 3:
        target_size = image_feats_list[0].shape[-2:]
    else:
        target_size = None

    flattened_feats_list = []
    for feats in image_feats_list:
        flattened_feats_list.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats_list, dim=0)   # (N, C)

    # ==[reduce]==
    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_reducer is None:
        if method.lower() == 'pca':
            fit_reducer = TorchPCA(n_components=dim, **kwargs).fit(x)
        elif method.lower() == 'umap':
            fit_reducer = TorchUMAP(n_components=dim, **kwargs).fit(x)
        elif method.lower() in ['tsne', 't-sne']:
            fit_reducer = TorchTSNE(n_components=dim, **kwargs).fit(x)
        else:
            raise ValueError(f"Unknown method: {method}. Choose 'pca', 'umap', or 'tsne'('t-sne')")

    # ==[transform]==
    flattened_feats_list = []
    reduced_feats = []
    if method.lower() in ['tsne', 't-sne']:
        for feats in image_feats_list:
            flattened_feats_list.append(flatten(feats))
        y = torch.cat(flattened_feats_list, dim=0)   # (N, C)

        # transform
        y_reduced = fit_reducer.transform(y)
        if isinstance(y_reduced, np.ndarray):
            y_reduced = torch.from_numpy(y_reduced)

        # Normalize to [0, 1] range per channel
        y_max, y_min = y_reduced.max(dim=0, keepdim=True).values, y_reduced.min(dim=0, keepdim=True).values
        if normalize:
            y_reduced = (y_reduced - y_min) / y_max

        # cut
        acc_len = 0
        for i, flattened_feats in enumerate(flattened_feats_list):
            cur_len = len(flattened_feats)
            if i == len(image_feats_list) - 1:
                reduced_feats.append(y_reduced[acc_len:])
            else:
                reduced_feats.append(y_reduced[acc_len:acc_len+cur_len])
            acc_len += cur_len

        # unflatten
        for i, y_reduced in enumerate(reduced_feats):
            y_reduced = unflatten(y_reduced, image_feats_list[i]).to(device)
            reduced_feats[i] = y_reduced
    else:
        # transform
        for feats in image_feats_list:
            y_flat = flatten(feats)
            y_reduced = fit_reducer.transform(y_flat)
            if isinstance(y_reduced, np.ndarray):
                y_reduced = torch.from_numpy(y_reduced)
            reduced_feats.append(y_reduced)
        y_reduced = torch.cat(reduced_feats, dim=0)

        # normalize & unflatten
        y_max, y_min = y_reduced.max(dim=0, keepdim=True).values, y_reduced.min(dim=0, keepdim=True).values
        for i, y_reduced in enumerate(reduced_feats):
            if normalize:
                y_reduced = (y_reduced - y_min) / y_max

            y_reduced = unflatten(y_reduced, image_feats_list[i]).to(device)
            reduced_feats[i] = y_reduced

    return reduced_feats, fit_reducer


# Alias functions for backward compatibility
def pca_reduce(image_feats_list: list, dim=3, fit_pca=None, max_samples=None):
    return reduce_dimensionality(image_feats_list, dim, fit_pca, method='pca', max_samples=max_samples)


def umap_reduce(image_feats_list: list, dim=3, fit_umap=None, max_samples=None, **kwargs):
    return reduce_dimensionality(image_feats_list, dim, fit_umap, method='umap', max_samples=max_samples, **kwargs)


def tsne_reduce(image_feats_list: list, dim=3, fit_tsne=None, max_samples=None, **kwargs):
    return reduce_dimensionality(image_feats_list, dim, fit_tsne, method='tsne', max_samples=max_samples, **kwargs)
#endregion


#region ==[Colorbar]==
def generate_tab10_saturation_gradients(n_levels=256):
    """
    Generate saturation gradients for each color in the Tab10 colormap.
    Each gradient goes from low saturation (near white) to the original color.

    Args:
        n_levels (int): Number of samples in each gradient.

    Returns:
        gradients (np.ndarray): Array of shape (10, n_levels, 3), containing RGB gradients for all 10 Tab10 colors.
    """
    # Choose palette (modified from DINOv3: https://github.com/facebookresearch/dinov3/blob/main/notebooks/segmentation_tracking.ipynb)
    if n_levels <= 10:
        color_map = cm.get_cmap("tab10")
        base_colors = [color_map(i)[:3] for i in range(10)]
    elif n_levels <= 20:
        color_map = cm.get_cmap("tab20")
        base_colors = [color_map(i)[:3] for i in range(20)]
    else:
        color_map = cm.get_cmap("gist_rainbow")
        base_colors = [color_map(i / (n_levels - 1))[:3] for i in range(n_levels)]

    gradients = []
    for color in base_colors:
        black = np.zeros(3)
        white = np.ones(3)
        gradient = np.linspace(white, black, n_levels + 1)

        color = np.array(color)
        gradient = gradient * white + (1 - gradient) * color
        gradients.append(gradient[1:])

    return gradients  # 10*(n_levels, 3)
#endregion

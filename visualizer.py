from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
import numpy as np



class LineStyle:
    styles = ['-', '--', ':', '-.']  # 4种线型
    
    @classmethod
    def get_style(cls, index):
        return cls.styles[index % len(cls.styles)]

class Marker:
    markers = ['o', '<', 's', '^', 'D', 'v', '>', 'X']  # 8种标记
    markersizes = [1, 1.2, 0.8, 1.2, 0.8, 1.2, 1.2, 1]  # 8种标记
    
    @classmethod
    def get_marker(cls, index):
        return cls.markers[index % len(cls.markers)]
    @classmethod
    def get_markersize(cls, index):
        return cls.markersizes[index % len(cls.markersizes)]

class Color:
    colors = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e',
        '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
        '#bcbd22', '#17becf'  # 10种颜色
    ]
    
    @classmethod
    def get_color(cls, index):
        return cls.colors[index % len(cls.colors)]

class StyleManager:
    @staticmethod
    def get_style(index, repeat=None, cycle=None):
        index_l = index
        index_m = index
        index_c = index
        if repeat is not None:
            index_l = index // repeat[0]
            index_m = index // repeat[1]
            index_c = index // repeat[2]
        if cycle is not None:
            index_l %= cycle[0]
            index_m %= cycle[1]
            index_c %= cycle[2]
        return {
            'linestyle': LineStyle.get_style(index_l),
            'marker': Marker.get_marker(index_m),
            'markersize': Marker.get_markersize(index_m),
            'color': Color.get_color(index_c),
            'linewidth': 1.75 + (index // 10) * 0.5  # 每10条线增加线宽
        }


def plot_subplots(data_list, titles, xlabels, ylabels, layout=(1, 1), subfig_size=(7, 5), grid=True, fontsize=12, title_fontsize=18, subtitle_fontsize=14,
                  top=0.95, bottom=0.1, left=0.1, right=0.95, hspace=0.25, wspace=0.15):
    """
    绘制多个子图的通用函数。
    
    参数:
    - data_list: 包含每个子图数据的列表，每个元素是一个元组 (x_data, y_data)。
    - titles: 每个子图的标题列表。
    - xlabels: 每个子图的 X 轴标签列表。
    - ylabels: 每个子图的 Y 轴标签列表。
    - layout: 子图的行数和列数。
    - grid: 是否显示网格。
    - fontsize: 轴标签和刻度的字体大小。
    - title_fontsize: 子图标题的字体大小。
    - subtitle_fontsize: 子图下方标签的字体大小。
    - top, bottom, left, right, hspace, wspace: 调整子图布局的参数。
    """
    # 计算子图的行列数
    num_plots = len(data_list)
    rows, cols = layout
    
    # 自动确定 figsize
    base_width, base_height = subfig_size 
    figsize = (cols * base_width, rows * base_height)

    # 创建子图
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()  # 将二维的 axes 数组展平为一维
    
    # 子图标签
    labels = [chr(97 + i) for i in range(num_plots)]  # 生成 a, b, c, d, e, f 等标签

    for i, (x_data, y_data) in enumerate(data_list):
        ax = axes[i]
        ax.scatter(x_data, y_data, s=5, alpha=0.5, zorder=1000)
        ax.set_xlabel(xlabels[i], fontsize=fontsize)
        ax.set_ylabel(ylabels[i], fontsize=fontsize)
        if titles[i] is not None:
            ax.set_title(titles[i], fontsize=title_fontsize) 
        ax.grid(linestyle='--', zorder=0)
        ax.tick_params(axis='both', labelsize=fontsize - 2)
        ax.text(0.5, -0.2, f"({labels[i]})", transform=ax.transAxes, fontsize=subtitle_fontsize, ha='center')

    # 隐藏多余的子图
    for j in range(num_plots, rows * cols):
        fig.delaxes(axes[j])

    # 调整布局
    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
    plt.show()
    

def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.01, y_ratio=0.01):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    x_range = x[zone_right] - x[zone_left]
    xlim_left = x[zone_left] - x_range * x_ratio
    xlim_right = x[zone_right] + x_range * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right + 1] for yi in y])
    y_range = np.max(y_data) - np.min(y_data)
    ylim_bottom = np.min(y_data) - y_range * y_ratio
    ylim_top = np.max(y_data) + y_range * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black", linewidth=1)

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax, linewidth=1)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax, linewidth=1)
    axins.add_artist(con)




from typing import Tuple, Optional, Dict, Union
import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from collections.abc import Iterable

from .config import cfg


for group, params in cfg['rc_params'].items():
    plt.rc(group, **params)



_ROOT = cfg['root']


def style_env(style: str):
    def decorator(func):
        def wrapper(*arg, **kwargs):
            with plt.style.context(cfg.default_style + style, after_reset=cfg.reset):
                results = func(*arg, **kwargs)
            return results
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    return decorator


class FreePlot:
    """
    A simple implement is used to draw some easy figures in my sense. 
    It is actually a rewrite based on matplotlib and seaborn as the former 
    is flexible but difficult to use and the latter is eaiser but not flexible.
    Therefore, I try my best to combine the both to make it easy to draw.
    At least, in my opinion, it's helpful.
    """
    def __init__(
        self, 
        shape: Tuple[int, int], 
        figsize: Tuple[float, float], 
        titles: Optional[Iterable]=None,
        **kwargs: "other kwargs of plt.subplots"
    ):
        """
        If you are familiar with plt.subplots, you will find most of 
        kwargs can be used here directly except
        titles: a list or tuple including the subtitles for differents axes.
        You can ignore this argument and we will assign (a), (b) ... as a default setting.
        Titles will be useful if you want call a axe by the subtitles or endowing the axes 
        different titles together.
        """
        self.root = _ROOT
        self.fig, axs = plt.subplots(figsize=figsize, nrows=shape[0], ncols=shape[1], **kwargs)
        self.axes = np.array(axs).flatten()
        self.titles = self.initialize_titles(titles)
    
    def initialize_titles(self, titles: Optional[Iterable]) -> Dict:
        n = len(self.axes)
        names = dict()
        if titles is None:
            for i in range(n):
                s = "(" + chr(i + 97) + ")"
                names.update({s:i})
        else:
            for i in range(n):
                title = titles[i]
                names.update({title:i})
        return names

    def legend(
        self, 
        x: float, y: float, ncol: int, 
        index: Union[int, str] = 0, 
        loc: str = "lower left"
    ) -> None:
        # tracky, fig[index].legend() is almost enough
        self[index].legend(bbox_to_anchor=(x, y),
        bbox_transform=plt.gcf().transFigure, ncol=ncol)

    def subplots_adjust(
        self,
        left: Optional[float] = None, 
        bottom: Optional[float] = None, 
        right: Optional[float] = None, 
        top: Optional[float] = None, 
        wspace: Optional[float] = None, 
        hspace: Optional[float] = None
    ) -> None:
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)

    def savefig(
        self, filename: str, 
        bbox_inches: str = 'tight', 
        tight_layout: bool = True,
        **kwargs: "other kwargs of plg.savefig"
    ) -> None:
        if tight_layout:
            plt.tight_layout()
        plt.savefig(
            self.root + filename,
            bbox_inches=bbox_inches,
            **kwargs
        )

    def set(
        self, 
        index: Union[int, str, Iterable, None] = None, 
        **kwargs: "other kwargs of ax.set"
    ) -> None:
        axes = self._extend_index(index)
        for ax in axes:
            ax.set(**kwargs)

    def set_titles(self, y: float = -0.3) -> None:
        for title in self.titles:
            ax = self[title]
            ax.set_title(title, y=y)

    def show(self, tight_layout: bool = True) -> None:
        if tight_layout:
            plt.tight_layout()
        plt.show()

    def _bound(
        self, 
        values: np.ndarray, 
        lower_bound: float, 
        upper_bound: float, 
        nums: int, 
        need_max: bool = False,
        need_min: bool = False
    ) -> np.ndarray:

        flag1 = values >= lower_bound
        flag2 = values <= upper_bound
        values = values[flag1 & flag2]
        if not need_min:
            lower_bound = values.min()
        if not need_max:
            upper_bound = values.max()
        return np.linspace(lower_bound, upper_bound, nums)
        
    def set_xticklabels(
        self, 
        lower_bound: float = 0., 
        upper_bound: float = 1., 
        index: Union[int, str, Iterable, None] = None,
        nums: int = 5, 
        format: float = "%.2f",
        need_min: bool = False,
        need_max: bool = True
    ) -> None:

        axes = self._extend_index(index)
        for ax in axes:
            values = ax.get_xticks()
            values = self._bound(values, lower_bound, upper_bound, nums,
                            need_min=need_min, need_max=need_max)
            labels = [format%value for value in values]
            ax.set_xticks(values)
            ax.set_xticklabels(labels)

    def set_yticklabels(
        self, 
        lower_bound: float = 0., 
        upper_bound: float = 1., 
        index: Union[int, str, Iterable, None] = None,
        nums: int = 5, 
        format: str = "%.2f",
        need_min: bool = False,
        need_max: bool = False
    ) -> None:

        axes = self._extend_index(index)
        for ax in axes:
            values = ax.get_yticks()
            values = self._bound(values, lower_bound, upper_bound, nums,
                            need_min=need_min, need_max=need_max)
            labels = [format%value for value in values]
            ax.set_yticks(values)
            ax.set_yticklabels(labels)

    @style_env(cfg.heatmap_style)
    def heatmap(
        self, data: pd.DataFrame, 
        index: Union[int, str] = 0, 
        annot: bool = True, 
        format: str = ".4f",
        cmap: str = 'GnBu', 
        linewidth: float = .5,
        **kwargs: "other kwargs of sns.heatmap"
    ) -> None:
        """
        data: M x N dataframe.
        cmap: GnBu, Oranges are recommanded.
        annot: annotation.
        fmt: the format for annotation.
        kwargs:
            cbar: bool
        """
        ax = self[index]
        sns.heatmap(
            data, ax=ax, 
            annot=annot, fmt=format,
            cmap=cmap, linewidth=linewidth,
            **kwargs
        )

    @style_env(cfg.lineplot_style)
    def lineplot(
        self, x: np.ndarray, y: np.ndarray, 
        index: Union[int, str] = 0, 
        seaborn: bool = False, 
        **kwargs: "other kwargs of ax.plot or sns.lineplot"
    ) -> None:
        ax = self[index]
        if seaborn:
            sns.lineplot(x, y, ax=ax, **kwargs)
        else:
            ax.plot(x, y, **kwargs)
        
    @style_env(cfg.scatterplot_style)
    def scatterplot(
        self, x: np.ndarray, y: np.ndarray, 
        index: Union[int, str] = 0, 
        seaborn: bool = False, 
        **kwargs: "other kwargs of ax.scatter or sns.scatterplot"
    ) -> None:
        ax = self[index]
        if seaborn:
            sns.scatterplot(x, y, ax=ax, **kwargs)
        else:
            ax.scatter(x, y, **kwargs)

    @style_env(cfg.imageplot_style)
    def imageplot(
        self, img: np.ndarray, 
        index: Union[int, str]=0, 
        show_ticks: bool = False, 
        **kwargs: "other kwargs of ax.imshow"
    ) -> None:
        ax = self[index]
        try:
            assert img.shape[2] == 3
            ax.imshow(img, **kwargs)
        except AssertionError:
            ax.imshow(img.squeeze(), cmap="gray", **kwargs)
        if not show_ticks:
            ax.set(xticks=[], yticks=[])

    @style_env(cfg.barplot_style)
    def barplot(
        self, x: str, y: str, hue: str, 
        data: pd.DataFrame, 
        index: Union[int, str] = 0, 
        **kwargs: "other kwargs of sns.barplot"
    ) -> None:
        ax = self[index]
        sns.barplot(x=x, y=y, hue=hue, data=data, ax=ax, **kwargs)
        self.fig.autofmt_xdate()

    def _extend_index(self, index: Union[int, str, Iterable, None] = None) -> Iterable:
        if index is None:
            return self.axes
        if not isinstance(index, (list, tuple)):
            index = [index]
        axes = self[index]
        return axes   

    def __getitem__(self, index: Union[int, str, Iterable]) -> "ax or [axes]":
        if isinstance(index, (list, tuple)):
            ax = []
            for i in index:
                try:
                    ind = self.titles[i]
                except KeyError:
                    ind = i
                finally:
                    ax.append(self.axes[ind])
            return ax
        else:
            try:
                ind = self.titles[index]
            except KeyError:
                ind = index
            finally:
                return self.axes[ind]



 
    

    
        





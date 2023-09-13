from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
from IPython import get_ipython  # type: ignore

import math
import json
import glob
import numpy as np
import pandas as pd
import scipy.stats as st
from datetime import datetime
import matplotlib.pyplot as plt

# Golden Ratio - https://en.wikipedia.org/wiki/Golden_ratio - 1.61803398875
GR = (1+5**0.5)/2

# Root path for all data
ROOT_PATH = Path.home() / "Code" / "YTA" / "1. Channel Analysis"

# Used typings
HASH = Dict[str, Any]


def wilson_score(pos: int, n: int, conf: float = 0.95):
    """
    Find the Wilson score for a binomial distribution.
    Wilson score is a confidence interval for a binomial distribution. It is
    symmetric around the mean and asymptotically approaches the normal.

    It is used to find the confidence interval for the proportion of positive
    ratings given a number of positive and negative/total ratings.

    Read more: https://stackoverflow.com/a/45965534
    Read more: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval
    """
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - conf) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def minmax(srs: Any):
    return (srs - srs.min()) / (srs.max() - srs.min())


class VideoInfoDownloader:
    def __init__(self, channel: str, root: Path = ROOT_PATH):
        self.root = root
        self.channel = channel

    @property
    def channel_path(self):
        return self.root.joinpath(self.channel)

    @property
    def video_info_path(self):
        return self.channel_path.joinpath("raw", "videos")

    @property
    def files(self):
        return self.video_info_path.glob("*.info.json")

    def run(self, force: bool = False):
        count = len(list(self.files))
        if count and not force:
            print(f"Found {count} info files, skipping download")
            return self

        cmd = "yt-dlp --write-info-json --skip-download --ignore-errors"
        cmd = f"{cmd} -o '%(id)s.%(ext)s' https://www.youtube.com/channel/{self.channel}/videos"
        cmd = "mkdir -p {self.video_info_path} && cd {self.video_info_path} && {cmd}"
        # run shell command
        get_ipython().run_line_magic("shell", cmd)
        return self


class VideoStatsProvider:
    def __init__(self, info_path: str):
        self.path = info_path
        self._info = None

    @property
    def info(self) -> Optional[HASH]:
        if self._info is None:
            with open(self.path, "r") as f:
                self._info = json.load(f)
        if self._info is None or 'upload_date' not in self._info:
            return None
        return self._info

    @property
    def engagement_score(self):
        if self.info is None:
            return 0

        likes = self.info.get('like_count', 0)
        comments = self.info.get('comment_count', 0)
        views = self.info.get('view_count', 0)

        like_ness = wilson_score(likes, views)
        comment_ness = wilson_score(comments, views)
        superfan_ness = wilson_score(comments, likes)

        return (like_ness * comment_ness * superfan_ness**GR)**(1/(2 + GR))

    @property
    def retention_score(self):
        if self.info is None or 'heatmap' not in self.info:
            return 0

        total_time, weighted_sum = 0, 0
        for segment in self.info['heatmap']:
            duration = segment['end_time'] - segment['start_time']
            total_time += duration

            weight = segment['end_time'] / total_time  # skew towards the end
            weighted_sum += duration * segment['value'] * weight

        return weighted_sum/total_time


class ChannelVideosRanker:
    keys = ['id', 'title', 'duration',
            'view_count', 'like_count', 'comment_count']

    def __init__(self, path: str, time_decay: float = 0.999, view_decay: float = 0.999, boost: float = 0.1):
        self.path = path
        self.boost = boost
        self.time_decay = time_decay
        self.view_decay = view_decay

    def sanitize_item(self, v: VideoStatsProvider, popular_views: float = 1e7):
        assert v.info is not None
        data = {k: v.info.get(k, 0) for k in self.keys}
        # data['id'] = f"https://www.youtube.com/watch?v={data['id']}"

        # calculate time decay for each video
        data['upload_date'] = datetime.strptime(
            v.info['upload_date'], "%Y%m%d")
        data['days_since'] = (
            datetime.now() - data['upload_date']).days
        data['decay_factor'] = self.time_decay**data['days_since']

        # boost engagement and retention for longer videos
        data['engagement'] = (
            v.engagement_score * v.info['duration'] ** self.boost)
        data['retention'] = (
            v.retention_score * v.info['duration'] ** self.boost)

        # calculate popularity for each video based on views and time delay
        data['normalized_view_count'] = data['view_count'] * data['decay_factor']
        data['acceptance'] = np.log(data['normalized_view_count'])
        popularity = self.view_decay ** (popular_views /
                                         data['normalized_view_count'])
        data['undervalued'] = (1 - popularity)**self.boost

        return data

    def run(self):
        videos = [VideoStatsProvider(fp)
                  for fp in glob.glob(f"{self.path}/*.info.json")]
        videos = [v for v in videos if v.info is not None]

        self.df = pd.DataFrame.from_records(
            [self.sanitize_item(v) for v in videos])
        self.df = self.df.sort_values(by='upload_date', ascending=True)
        self.df = self.df.set_index('upload_date')

        return self

    def find_undervalued_videos(self):
        fields = ['engagement', 'retention', 'undervalued', 'acceptance']

        self.df['score'] = 1
        for field in fields:
            self.df[field] /= self.df[field].max()
            self.df['score'] *= self.df[field]

        self.df = self.df[self.df['score'] > 0]

        cols = ['id', 'title', 'duration', 'view_count', 'score']
        self.df = self.df.sort_values(by='score', ascending=False)
        return self.df[cols+fields]

    def scatter_view(self, fn: Optional[Any] = None, label: str = 'engagement'):
        _fig, ax = plt.subplots(2, 2, figsize=(12, 6))

        xf = fn(self.df) if fn else self.df
        ax[0][0].scatter(xf.index, xf[label])
        ax[0][1].scatter(np.log10(xf['view_count']), xf[label])  # type: ignore

        x = xf['retention'] if label == 'score' else xf['score']
        ax[1][1].scatter(x, xf[label])
        ax[1][0].scatter(xf['duration']/3600, xf[label])

import numpy as np
import pandas as pd
import sys
import timeit
import warnings
from itertools import chain
from typing import Callable, Dict, List, Optional, Iterable, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy import sparse
# from hierarchicalforecast.units import aggregate

from utils.forecast_prep import forecast_prep
from utils.data_process import get_unique_id


def tarin_predict(
		df: pd.DataFrame,
		grp: List,
		model: object,
		spec: List,
		exclude: Tuple = ('period',),
		horizon: int = 5
) -> pd.DataFrame:
	"""

	:param df:
	:param grp:
	:param model:
	:param spec:
	:param exclude:
	:param horizon:
	:return:
	"""
	# Split the data into different levels
	train = forecast_prep(
		df=df,
		split_date=None,
		grp=grp,
	)
	train_agg, S_train, tags = aggregate(train, spec)
	model.fit(df=train_agg.reset_index())

	preds = model.predict(horizon)

	_cols = [c for c in grp if c not in exclude]
	preds[_cols] = (
		preds['unique_id']
		.str
		.split('/', n=len(_cols), expand=True)
	)
	preds = preds.dropna().reset_index(drop=True)

	return preds


def get_detalization(
		df: pd.DataFrame,
		answer: pd.DataFrame,
		model_name: str,
		uid: List,
		grp: List,
		exclude: Tuple = ('period',),
		start_period: str = '2023-01-01',
		weight_factor: Union[int, float] = 60
) -> pd.DataFrame:
	"""

	:param df:
	:param answer:
	:param model_name:
	:param uid:
	:param grp:
	:param exclude:
	:param start_period:
	:param weight_factor:
	:return:
	"""
	_cols = [c for c in grp if c not in exclude]
	# Add unique detailed TS id.
	df_mp_uid = get_unique_id(df=df, uid=uid)
	df_mp_uid_full = (
		df_mp_uid[
			(df_mp_uid['period'] >= start_period)
			& (df_mp_uid.client_sap_id != -1)
			]
		.reset_index(drop=True)
		.rename(columns={'unique_ts': 'unique_ts_full'})
	)
	df_mp_uid_full.holding_name = df_mp_uid_full.holding_name.astype(np.int64)
	# Add unique less detailed TS id (the one used for prediction).
	df_mp_uid_less = get_unique_id(df=df_mp_uid_full, uid=_cols)
	# Calculate fraction of low level TS inside high level TS.
	df_calc = (
		df_mp_uid_less.groupby(['unique_ts', 'unique_ts_full'])
		.unique_ts_full
		.count()
		.to_frame()
		.rename(columns={'unique_ts_full': 'count'})
		.reset_index()
	)
	df_calc['tot_count'] = df_calc.groupby('unique_ts')['count'].transform('sum')
	df_calc['perc'] = df_calc['count']/df_calc['tot_count']

	# Formatting fina, dataframe
	df_calc[_cols] = (
		df_calc['unique_ts']
		.str
		.split('_', n=len(_cols), expand=True)
	)
	sub_answer = answer.merge(
		df_calc[_cols + ['perc', 'unique_ts_full']],
		how='left',
		on=_cols
	).dropna().reset_index(drop=True)
	sub_answer_lgb = sub_answer[['ds', model_name, 'perc', 'unique_ts_full']]
	sub_answer_lgb['forecast_wagon_count'] = sub_answer_lgb[model_name] * sub_answer_lgb['perc']
	sub_answer_lgb['forecast_wagon_count'] = sub_answer_lgb['forecast_wagon_count'].round().astype(np.int64)
	sub_answer_lgb = sub_answer_lgb.rename(columns={'ds': 'period'}).drop(columns=[model_name, 'perc'])
	sub_answer_lgb[uid] = (
		sub_answer_lgb['unique_ts_full']
		.str
		.split('_', n=len(uid), expand=True)
	)
	sub_answer_lgb = sub_answer_lgb.drop(columns=['unique_ts_full'])[['period'] + uid + ['forecast_wagon_count']]
	sub_answer_lgb['forecast_weight'] = sub_answer_lgb['forecast_wagon_count'] * weight_factor
	sub_answer_lgb['forecast_weight'] = sub_answer_lgb['forecast_weight'].astype(np.float64)

	_order = [
		'period',
		'rps',
		'podrod',
		'filial',
		'client_sap_id',
		'freight_id',
		'sender_station_id',
		'recipient_station_id',
		'sender_organisation_id',
		'forecast_weight',
		'forecast_wagon_count'
	]

	return sub_answer_lgb[_order]




def _to_upper_hierarchy(bottom_split, bottom_values, upper_key):
    upper_split = upper_key.split('/')
    upper_idxs = [bottom_split.index(i) for i in upper_split]

    def join_upper(bottom_value):
        bottom_parts = bottom_value.split('/')
        return '/'.join(bottom_parts[i] for i in upper_idxs)

    return [join_upper(val) for val in bottom_values]

# %% ../nbs/utils.ipynb 12
def aggregate(
    df: pd.DataFrame,
    spec: List[List[str]],
    is_balanced: bool = False,
    sparse_s: bool = False,
):
    """Utils Aggregation Function.
    Aggregates bottom level series contained in the pandas DataFrame `df` according
    to levels defined in the `spec` list.

    Parameters
    ----------
    df : pandas DataFrame
        Dataframe with columns `['ds', 'y']` and columns to aggregate.
    spec : list of list of str
        List of levels. Each element of the list should contain a list of columns of `df` to aggregate.
    is_balanced : bool (default=False)
        Deprecated.
    sparse_s : bool (default=False)
        Return `S_df` as a sparse dataframe.

    Returns
    -------
    Y_df : pandas DataFrame
        Hierarchically structured series.
    S_df : pandas DataFrame
        Summing dataframe.
    tags : dict
        Aggregation indices.
    """
    # Checks
    if df.isnull().values.any():
        raise ValueError('`df` contains null values')
    if is_balanced:
        warnings.warn(
            "`is_balanced` is deprecated and will be removed in a future version. "
            "Don't set this argument to suppress this warning.",
            category=DeprecationWarning,
        )
            
    # compute aggregations and tags
    spec = sorted(spec, key=len)
    bottom = spec[-1]
    aggs = []
    tags = {}
    for levels in spec:
        agg = df.groupby(levels + ['ds'])['y'].sum().reset_index('ds')
        group = agg.index.get_level_values(0)
        for level in levels[1:]:
            group = group + '/' + agg.index.get_level_values(level).str.replace('/', '_')
        agg.index = group
        agg.index.name = 'unique_id'
        tags['/'.join(levels)] = group.unique().values
        aggs.append(agg)
    Y_df = pd.concat(aggs)

    # construct S
    bottom_key = '/'.join(bottom)
    bottom_levels = tags[bottom_key]
    S = np.empty((len(bottom_levels), len(spec)), dtype=object)
    for j, levels in enumerate(spec[:-1]):
        S[:, j] = _to_upper_hierarchy(bottom, bottom_levels, '/'.join(levels))
    S[:, -1] = tags[bottom_key]
    categories = list(tags.values())
    try:
        encoder = OneHotEncoder(categories=categories, sparse_output=sparse_s, dtype=np.float32)
    except TypeError:  # sklearn < 1.2
        encoder = OneHotEncoder(categories=categories, sparse=sparse_s, dtype=np.float32)    
    S = encoder.fit_transform(S).T
    if sparse_s:
        df_constructor = pd.DataFrame.sparse.from_spmatrix
    else:
        df_constructor = pd.DataFrame
    S_df = df_constructor(S, index=np.hstack(categories), columns=bottom_levels)
    return Y_df, S_df, tags

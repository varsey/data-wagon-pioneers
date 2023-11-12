import pandas as pd
from typing import List, Union, Tuple, Optional


def forecast_prep(
		df: pd.DataFrame,
		split_date: Optional[str],
		grp: List,
		exclude: Tuple = ('period', ),
) -> Union[Tuple, pd.DataFrame]:
	"""
	Подготовка данных к обучению + Train/Test Split.
	:param df: input dataframe
	:param split_date: date to split the data
	:param grp: list of column names to group by
	:param exclude: column name to exclude from grp (date column)
	:return: training-ready data
	"""
	_cols = [c for c in grp if c not in exclude]
	# Format data.
	hf = df.rename(columns={'period': 'ds', 'real_wagon_count': 'y'})
	hf[_cols] = hf[_cols].astype(str)
	# Get train and validation datasets
	if split_date:
		train = hf.loc[hf['ds'] <= split_date]
		valid = hf.loc[hf['ds'] > split_date]
		return train, valid
	else:
		return hf

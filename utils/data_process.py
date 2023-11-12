import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import List, Tuple


def add_master_data_mappings(df: pd.DataFrame, data_folder: Path) -> pd.DataFrame:
    # = Пути к справочникам - откорректировать если в реальной системе будут лежать по другому адресу =
    client_mapping_file = data_folder / "client_mapping.csv"
    freight_mapping_file = data_folder / "freight_mapping.csv"
    station_mapping_file = data_folder / "station_mapping.csv"
    # Клиент - холдинг
    client_mapping = pd.read_csv(
        client_mapping_file,
        sep=";",
        decimal=",",
        encoding="windows-1251",
    )
    df = pd.merge(df, client_mapping, how="left", on="client_sap_id")
    # Груз
    freight_mapping = pd.read_csv(
        freight_mapping_file, sep=";", decimal=",", encoding="windows-1251"
    )
    df = pd.merge(df, freight_mapping, how="left", on="freight_id")
    # Станции
    station_mapping = pd.read_csv(
        station_mapping_file,
        sep=";",
        decimal=",",
        encoding="windows-1251",
    )
    df = pd.merge(
        df,
        station_mapping.add_prefix("sender_"),
        how="left",
        on="sender_station_id",
    )
    df = pd.merge(
        df,
        station_mapping.add_prefix("recipient_"),
        how="left",
        on="recipient_station_id",
    )
    return df


def evaluate(fact: pd.DataFrame, forecast: pd.DataFrame, data_folder: Path, public: bool = True) -> float:
    # = Параметры для расчета метрики =
    accuracy_granularity = [
        "period",
        "rps",
        "holding_name",
        "sender_department_name",
        "recipient_department_name",
    ]
    fact_value, forecast_value = "real_wagon_count", "forecast_wagon_count"
    if public:
        metric_weight = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
    else:
        metric_weight = np.array([0.1, 0.6, 0.1, 0.1, 0.1])

    # = Собственно расчет метрик =
    # 1. Добавляем сущности верхних уровней гранулярности по справочникам
    fact = add_master_data_mappings(fact, data_folder)
    forecast = add_master_data_mappings(forecast, data_folder)

    # 2. Расчет KPI
    compare_data = pd.merge(
        fact.groupby(accuracy_granularity, as_index=False)[fact_value].sum(),
        forecast.groupby(accuracy_granularity, as_index=False)[forecast_value].sum(),
        how="outer",
        on=accuracy_granularity,
    ).fillna(0)
    # Против самых хитрых - нецелочисленный прогноз вагоноотправок не принимаем
    compare_data[fact_value] = np.around(compare_data[fact_value]).astype(int)
    compare_data[forecast_value] = np.around(compare_data[forecast_value]).astype(int)

    # 3. Рассчитаем метрики для каждого месяца в выборке
    compare_data["ABS_ERR"] = abs(
        compare_data[forecast_value] - compare_data[fact_value]
    )
    compare_data["MAX"] = abs(compare_data[[forecast_value, fact_value]].max(axis=1))
    summary = compare_data.groupby("period")[
        [forecast_value, fact_value, "ABS_ERR", "MAX"]
    ].sum()
    summary["Forecast Accuracy"] = 1 - summary["ABS_ERR"] / summary["MAX"]

    # 4. Взвесим метрики отдельных месяцев для получения одной цифры score
    score = (
        summary["Forecast Accuracy"].sort_index(ascending=True) * metric_weight
    ).sum()

    return score


def calc_score_public(fact: pd.DataFrame, forecast: pd.DataFrame) -> float:
    return evaluate(fact, forecast, data_folder, public=True)


def calc_score_private(fact: pd.DataFrame, forecast: pd.DataFrame) -> float:
    return evaluate(fact, forecast, data_folder, public=False)


# Preprocess data.
def process_ts(
        df: pd.DataFrame,
        grp: List,
        target: str = 'real_wagon_count',
        cutoff: str = "20120101"
) -> pd.DataFrame:
    """
    Подготовка датасета для прогнозирования.
    :param df: input dataframe
    :param grp: list of column names to group by
    :param target: target column name
    :param cutoff: datetime threshold for cutting of older time periods
    :return: processed dataframe
    """
    _df = (
        df[grp + [target]]
        .query(f'period >= {cutoff}')
        .groupby(grp)[target]
        .sum()
        .reset_index()
    )
    return _df


# Generate unique id for initial data.
def get_unique_id(df: pd.DataFrame, uid: List) -> pd.DataFrame:
    """
    Генерация уникального id временного ряда.
    :param df: input dataframe
    :param uid: columns to concatenate
    :return: dataframe with ts unique ids column
    """
    pdf = pl.from_pandas(df)
    _df = (
        pdf.with_columns(
            pdf.select([
                pl.concat_str(
                    pl.col(uid), 
                    separator='_'
                )
                .alias("unique_ts"),
            ]))
        .to_pandas()
    )
    return _df


def ts_length(df: pd.DataFrame, grp: List, exclude: Tuple = ("period",)) -> pd.DataFrame:
    """
    Выравнивание временных рядов.
    :param df: input dataframe
    :param grp: list of column names to make unique id
    :param exclude: column name to exclude from grp (date column)
    :return: full-length dataframe
    """
    _cols = [c for c in grp if c not in exclude]
    # Get unique ids.
    df_mp_agg_ = get_unique_id(df, _cols).drop(columns=_cols)
    # Compute cartesian product.
    cartesian = pd.DataFrame(
        data=pd.MultiIndex.from_product([
            df_mp_agg_.period.unique(),
            df_mp_agg_.unique_ts.unique()
        ]).to_list(),
        columns=list(exclude) + ['unique_ts']
    )
    # Compile final dataframe.
    df_mp_agg_full = cartesian.merge(
        df_mp_agg_,
        how='left',
        on=list(exclude) + ['unique_ts']
    ).fillna(0)
    # Expand unique ids.
    df_mp_agg_full[_cols] = (
        df_mp_agg_full['unique_ts']
        .str
        .split('_', n=len(_cols), expand=True)
    )
    # Final shape.
    df_mp_agg_full = (
        df_mp_agg_full.drop(columns=['unique_ts'])
        .assign(real_wagon_count=lambda x: x['real_wagon_count'].astype(np.int64))
    )
    return df_mp_agg_full


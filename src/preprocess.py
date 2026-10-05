''' src/preprocess.py
- clean promo_feed.csv
- build client map
- check_status
- normalize
'''

import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)

pd.set_option("display.max_colwidth", 200)

project_root_dir = Path(__file__).resolve().parent.parent

def to_date(text):
    formats = [
        "%Y-%m-%d",    # 2026-10-01
        "%m/%d/%Y",    # 09/28/2026 (美式 月/日/年)
        "%d/%m/%Y",    # 01/10/2026 (日/月/年)
        "%d %b. %Y",   # 1 oct. 2026 / 7 oct. 2026
        "%b. %d %Y",   # oct. 1 2026
        "%d %B %Y",    # 7 octobre 2026
        "%Y-%m-%d %H:%M:%S"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass            # 这个格式不对，尝试下一个

    # 返回 None 会让 check_status 判成 need review，必须留痕否则无声丢失
    if text:
        logger.warning("日期解析失败，%r 不匹配任何已知格式", text)


def build_client_map():
    config_file = (project_root_dir / 'config' / 'clients.yaml')
    with open(config_file, 'r', encoding='utf-8') as f:
        clients = yaml.safe_load(f)

    logger.debug("clients.yaml: %s", clients)
    client_map = {}
    for key, cfg in clients.items():
        client_map[key.lower().strip()] = key
        client_map[cfg['display_name'].lower().strip()] = key
    logger.debug("client 别名映射: %s", client_map)
    return client_map


def clean_client(feeds_df):
    feeds_df['client'] = feeds_df['client'].str.strip().str.lower()
    logger.debug("归一化前的 client 分布:\n%s", feeds_df['client'].value_counts())

    client_map = build_client_map()
    feeds_df['client'] = feeds_df['client'].map(client_map)

    unknown = feeds_df[feeds_df['client'].isna()]
    if not unknown.empty:
        # 这些行会被丢掉：必须报出来，否则数据无声消失
        logger.warning("%s 行的 client 不在 clients.yaml 中，将被丢弃: %s",
                       len(unknown), unknown['sku'].tolist())
        logger.debug("被丢弃的行:\n%s", unknown)

    feeds_df = feeds_df.dropna(subset=['client'])
    logger.info("client 归一化后剩 %s 行:\n%s", len(feeds_df), feeds_df['client'].value_counts())
    return feeds_df


def load_flyer_period():
    with open(project_root_dir / 'config' / 'flyer_run.yaml', 'r', encoding='utf-8') as f:
        flyer = yaml.safe_load(f)
    
    return flyer["flyer_period"]["start"], flyer["flyer_period"]["end"]

FLYER_START, FLYER_END = load_flyer_period()


def check_status(row):
    start = row['promo_start']
    end = row['promo_end']
  
    # 1. 判断是否为空 (结合了 None, NaT, 空字符串)
    if pd.isna(start) or pd.isna(end) or start == '' or end == '':
      return 'need review'
  
    # 2. 比较日期大小
    if start > end:
      return 'need review'

    if start <= FLYER_END and end >= FLYER_START:
        return 'valid'
  
    return 'out_of_period'


def preprocess():
    src = project_root_dir / 'data' / 'promo_feed.csv'
    feeds_df = pd.read_csv(src, dtype=str, keep_default_na=False)
    logger.info("读入 %s：%s 行 × %s 列", src.name, *feeds_df.shape)
    logger.debug("前 5 行:\n%s", feeds_df.head())

    # deal with duplicates
    before = len(feeds_df)
    feeds_df = feeds_df.drop_duplicates(subset=['client','sku'], keep="first")
    logger.info("按 (client, sku) 去重: %s -> %s 行（丢弃 %s）",
                before, len(feeds_df), before - len(feeds_df))

    # deal with date
    logger.debug("promo_start 原始取值:\n%s", feeds_df['promo_start'].value_counts())
    logger.debug("promo_end 原始取值:\n%s", feeds_df['promo_end'].value_counts())
    feeds_df["promo_start"] = feeds_df["promo_start"].apply(to_date)
    feeds_df["promo_end"] = feeds_df["promo_end"].apply(to_date)

    # normalize client naming
    feeds_df = clean_client(feeds_df)

    # check status
    feeds_df['status'] = feeds_df.apply(check_status, axis=1)
    counts = feeds_df['status'].value_counts(dropna=False)
    logger.info("flyer 期 %s ~ %s，status 分布:\n%s", FLYER_START, FLYER_END, counts)
    if counts.get('need review', 0):
        logger.warning("%s 行日期有问题需人工确认", counts['need review'])
    logger.debug("清洗结果前 5 行:\n%s", feeds_df.head(5))

    out = project_root_dir / 'data' / 'promo_feed_clean.csv'
    feeds_df.to_csv(out, index=False)
    logger.info("已写出 %s（%s 行）", out.name, len(feeds_df))


if __name__=='__main__':
    from logging_setup import setup_logging

    setup_logging()
    preprocess()
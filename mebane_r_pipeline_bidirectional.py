#!/usr/bin/env python3
"""
mebane_r_pipeline_bidirectional.py

Loads Korean election results, builds the (leader_votes, total_votes,
eligible, province) unit tables required by the Ferrari/Mebane eforensics
"qbl" model, optionally replaces the data with a fraud-free synthetic null,
and fits one election/level/channel/source/leader_side cell via
mebane_crosscheck.R.

`leader_side` selects which party/candidate is coded as the eforensics
"leader" -- i.e. which side the model is asked to test for fraud in favor
of. eforensics' fraud classes (Manufactured votes via iota, Stolen votes via
chi) only ever add votes to the coded leader; the model cannot detect fraud
against the leader. 'dem' (default) codes the Democratic-lineage party or
candidate as leader; 'con' codes the conservative-lineage party or candidate
as leader, for the complementary direction of fraud test.

Usage (single cell):
  python3 mebane_r_pipeline_bidirectional.py <elec_id> <level> <channel> <source> \\
      [leader_side] [r_n_iter] [r_n_chains] [r_burn_in] [use_parcomp] [r_n_adapt]

  leader_side: 'dem' (default) or 'con'.

Election ids: 18, 19, 20, 21, 22 (general elections) or pres16..pres21
(presidential elections). Level: dong or constituency. Channel: depends on
level (see DONG_CHANNELS/CONST_CHANNELS below) -- 'total_no_early' for
elections with no early-voting split (see NO_EARLY_VOTING_ELECTIONS).
Source: real, marginal_null, or joint_null.

Example -- testing fraud in favor of the conservative side in a given cell:
  python3 mebane_r_pipeline_bidirectional.py pres18 dong total_no_early real con
  python3 mebane_r_pipeline_bidirectional.py pres18 dong total_no_early marginal_null con
  python3 mebane_r_pipeline_bidirectional.py pres18 dong total_no_early joint_null con
"""
import os
import re
import sys
import shutil
import subprocess
import tempfile
import time
import numpy as np
import pandas as pd

R_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mebane_crosscheck.R')
GENERAL_ELECTION_INT_KEYS = {'18', '19', '20', '21', '22'}
MIN_VOTES = 50


# ============================================================================
# Election configuration
# ============================================================================
ELECTION_CONFIGS = {
    'pres16': {
        'election_type':  'presidential',
        'result_csv':     '16th_presidential_election_result.csv',
        'dem_pattern':    r'노무현|새천년민주당',
        'con_pattern':    r'이회창|한나라당',
        'third_pattern':  r'이한동|권영길',
        'label':          '16th Presidential Election (2002)',
        'year':           2002,
        'election_month': 12,
        'no_early_voting': True,
    },
    'pres17': {
        'election_type':  'presidential',
        'result_csv':     '17th_presidential_election_result.csv',
        'dem_pattern':    r'정동영|대통합민주신당',
        'con_pattern':    r'이명박',
        'third_pattern':  r'이회창|권영길|이인제|문국현',
        'label':          '17th Presidential Election (2007)',
        'year':           2007,
        'election_month': 12,
        'no_early_voting': True,
    },
    'pres18': {
        'election_type':  'presidential',
        'result_csv':     '18th_presidential_election_result.csv',
        'dem_pattern':    r'문재인|민주통합당',
        'con_pattern':    r'박근혜|새누리당',
        'third_pattern':  None,
        'label':          '18th Presidential Election (2012)',
        'year':           2012,
        'election_month': 12,
        'no_early_voting': True,
    },
    'pres19': {
        'election_type':  'presidential',
        'result_csv':     '19th_presidential_election_result.csv',
        'dem_pattern':    r'더불어민주당',
        'con_pattern':    r'자유한국당',
        'third_pattern':  r'유승민|바른정당',
        'label':          '19th Presidential Election (2017)',
        'year':           2017,
        'election_month': 5,
    },
    21: {
        'election_type':  'general',
        'result_csv':     '21st_election_result.csv',
        'dem_pattern':    r'더불어민주당',
        'con_pattern':    r'미래통합당|자유한국당',
        'third_pattern':  None,
        'label':          '21st General Election (2020)',
        'year':           2020,
        'election_month': 4,
    },
    22: {
        'election_type':  'general',
        'result_csv':     '22nd_election_result.csv',
        'dem_pattern':    r'더불어민주당',
        'con_pattern':    r'국민의힘',
        'third_pattern':  r'개혁신당',
        'label':          '22nd General Election (2024)',
        'year':           2024,
        'election_month': 4,
    },
    18: {
        'election_type':   'general',
        'result_csv':      '18th_election_result.csv',
        'dem_pattern':     r'통합민주당',
        'con_pattern':     r'한나라당',
        'third_pattern':   None,
        'label':           '18th General Election (2008)',
        'year':            2008,
        'election_month':  4,
        'no_early_voting': True,
    },
    19: {
        'election_type':   'general',
        'result_csv':      '19th_election_result.csv',
        'dem_pattern':     r'민주통합당',
        'con_pattern':     r'새누리당',
        'third_pattern':   None,
        'label':           '19th General Election (2012)',
        'year':            2012,
        'election_month':  4,
        'no_early_voting': True,
    },
    20: {
        'election_type':   'general',
        'result_csv':      '20th_election_result.csv',
        'dem_pattern':     r'더불어민주당',
        'con_pattern':     r'새누리당',
        'third_pattern':   r'국민의당',
        'label':           '20th General Election (2016)',
        'year':            2016,
        'election_month':  4,
    },
    'pres20': {
        'election_type':  'presidential',
        'result_csv':     '20th_presidential_election_result.csv',
        'dem_pattern':    r'더불어민주당',
        'con_pattern':    r'국민의힘',
        'third_pattern':  None,
        'label':          '20th Presidential Election (2022)',
        'year':           2022,
        'election_month': 3,
    },
    'pres21': {
        'election_type':  'presidential',
        'result_csv':     '21st_presidential_election_result.csv',
        'dem_pattern':    r'더불어민주당',
        'con_pattern':    r'국민의힘',
        'third_pattern':  r'이준석|개혁신당',
        'label':          '21st Presidential Election (2025)',
        'year':           2025,
        'election_month': 6,
    },
}

SPECIAL_DONG_NAMES_GENERAL = {
    '거소·선상투표', '관외사전투표', '국외부재자투표',
    '국외부재자투표(공관)', '잘못 투입·구분된 투표지',
    '국내부재자투표',  # 19th General Election (2012)'s absentee-category naming variant
    '부재자투표',      # 18th General Election (2008)'s absentee-category naming variant
}
SPECIAL_DONG_NAMES_PRESIDENTIAL = {
    '거소·선상투표', '관외사전투표', '재외투표',
    '잘못 투입·구분된 투표지',
    '국내부재자투표',  # 18th presidential election (2012)'s absentee-category naming variant
    '부재자투표',      # 16th/17th presidential elections (2002/2007)'s naming variant
}
GWANNAESA_LABEL = '관내사전투표'
META_CANDIDATES = {'선거인수', '투표수', '무효 투표수', '기권자수'}

# Full-name and short-form Korean province/city names -> English short codes. Both forms are
# needed since which one appears in the raw data varies by election (older elections' raw CSVs
# sometimes only use the short do/province-level form).
PROV_FULL_TO_SHORT = {
    '서울특별시': 'Seoul',  '부산광역시': 'Busan',   '대구광역시': 'Daegu',
    '인천광역시': 'Incheon','광주광역시': 'Gwangju', '대전광역시': 'Daejeon',
    '울산광역시': 'Ulsan',  '세종특별자치시': 'Sejong',
    '경기도': 'Gyeonggi',  '강원도': 'Gangwon',     '강원특별자치도': 'Gangwon',
    '충청북도': 'Chungbuk', '충청남도': 'Chungnam',
    '전라북도': 'Jeonbuk',  '전북특별자치도': 'Jeonbuk', '전라남도': 'Jeonnam',
    '경상북도': 'Gyeongbuk','경상남도': 'Gyeongnam', '제주특별자치도': 'Jeju',
    '서울': 'Seoul',   '부산': 'Busan',    '대구': 'Daegu',
    '인천': 'Incheon', '광주': 'Gwangju',  '대전': 'Daejeon',
    '울산': 'Ulsan',   '세종': 'Sejong',
    '경기': 'Gyeonggi','강원': 'Gangwon',  '충북': 'Chungbuk',
    '충남': 'Chungnam','전북': 'Jeonbuk',  '전남': 'Jeonnam',
    '경북': 'Gyeongbuk','경남': 'Gyeongnam','제주': 'Jeju',
}


# ============================================================================
# Election-result CSV loading and per-unit vote aggregation
# ============================================================================

def _read_csv_auto(path: str, **kwargs) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding='utf-8', **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='cp949', **kwargs)


def normalize_dong_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    name = re.sub(r'\(.*?\)', '', name).strip().replace('.', '·')
    name = re.sub(r'제(\d)', r'\1', name)
    name = re.sub(r'·\d+', '', name)
    name = re.sub(r'(\d+)(동|읍|면)$', r'\2', name)
    return re.sub(r'\s+', ' ', name)


def split_admin_tokens(name: str) -> list:
    tokens, buf = [], []
    for ch in name:
        buf.append(ch)
        if ch in '시군구' and len(buf) >= 2:
            tokens.append(''.join(buf))
            buf = []
    if buf:
        tokens.append(''.join(buf))
    return [t for t in tokens if t]


def normalize_sigungu(name: str) -> list:
    if not isinstance(name, str):
        return []
    name = re.sub(r'\(.*?\)', '', name).strip()
    if not name:
        return []
    tokens = split_admin_tokens(name)
    if not tokens:
        stripped = re.sub(r'[시군구갑을병정무]$', '', name).strip()
        return [stripped] if stripped else []
    si_gun_count = sum(1 for t in tokens if t[-1] in '시군' and len(t) >= 2)
    gu_count = sum(1 for t in tokens if t[-1] == '구' and len(t) >= 2)
    ordered = tokens if (si_gun_count >= 2 or (si_gun_count == 0 and gu_count >= 2)) else list(reversed(tokens))
    candidates = []
    for t in ordered:
        key = re.sub(r'[시군구]$', '', t).strip()
        if key and key not in candidates:
            candidates.append(key)
    return candidates


def load_election_csv(csv_path: str, dem_pattern: str, con_pattern: str,
                       third_pattern: str = None, election_type: str = 'general'):
    try:
        df = _read_csv_auto(csv_path, low_memory=False)
    except Exception:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if election_type == 'presidential':
        df = df.rename(columns={'구시군명': '선거구명', '읍면동명': '법정읍면동명'})
        special_dong_names = SPECIAL_DONG_NAMES_PRESIDENTIAL
    else:
        special_dong_names = SPECIAL_DONG_NAMES_GENERAL

    df['득표수'] = pd.to_numeric(df['득표수'], errors='coerce').fillna(0).astype(int)
    df['is_dem'] = df['후보자'].str.contains(dem_pattern, case=False, na=False)

    if election_type == 'general':
        sejong_gap_kim = (df['선거구명'].astype(str).str.replace(' ', '').str.contains('세종.*갑', na=False)) & \
                         (df['후보자'].astype(str).str.contains('김종민', na=False))
        df.loc[sejong_gap_kim, 'is_dem'] = True

    df['is_con'] = df['후보자'].str.contains(con_pattern, case=False, na=False)
    df['is_third'] = df['후보자'].str.contains(third_pattern, case=False, na=False) if third_pattern else False
    df['is_meta'] = df['후보자'].isin(META_CANDIDATES)
    df['is_early'] = df['투표구명'] == GWANNAESA_LABEL

    dong_key = ['시도명', '선거구명', '법정읍면동명']
    const_key = ['시도명', '선거구명']

    def sgg_cands(name):
        if not isinstance(name, str):
            return []
        if '_' in name:
            return normalize_sigungu(name.split('_', 1)[1])
        return normalize_sigungu(re.sub(r'[갑을병정무]$', '', name).strip())

    df_geo = df[~df['법정읍면동명'].isin(special_dong_names)].copy()
    df_votes = df_geo[~df_geo['is_meta']].copy()

    gn_dem = df_votes[df_votes['is_dem'] & df_votes['is_early']].groupby(dong_key)['득표수'].sum().reset_index(name='in_precinct_early_dem')
    gn_con = df_votes[df_votes['is_con'] & df_votes['is_early']].groupby(dong_key)['득표수'].sum().reset_index(name='in_precinct_early_con')
    gn_tot = df_votes[df_votes['is_early']].groupby(dong_key)['득표수'].sum().reset_index(name='in_precinct_early_total')
    gn_third = df_votes[df_votes['is_third'] & df_votes['is_early']].groupby(dong_key)['득표수'].sum().reset_index(name='in_precinct_early_third')

    sd_dem = df_votes[df_votes['is_dem'] & ~df_votes['is_early']].groupby(dong_key)['득표수'].sum().reset_index(name='same_day_dem')
    sd_con = df_votes[df_votes['is_con'] & ~df_votes['is_early']].groupby(dong_key)['득표수'].sum().reset_index(name='same_day_con')
    sd_tot = df_votes[~df_votes['is_early']].groupby(dong_key)['득표수'].sum().reset_index(name='same_day_total')
    sd_third = df_votes[df_votes['is_third'] & ~df_votes['is_early']].groupby(dong_key)['득표수'].sum().reset_index(name='same_day_third')

    sum_people_dong = (df_geo[~df_geo['is_early'] & (df_geo['후보자'] == '선거인수')].groupby(dong_key)['득표수'].sum().reset_index(name='sum_people'))
    sum_vote_geo = (df_geo[df_geo['후보자'] == '투표수'].groupby(dong_key)['득표수'].sum().reset_index(name='sum_vote_geo'))

    df_dong = gn_dem.copy()
    for frame in (gn_con, gn_tot, gn_third, sd_dem, sd_con, sd_tot, sd_third, sum_people_dong, sum_vote_geo):
        df_dong = df_dong.merge(frame, on=dong_key, how='outer')
    df_dong = df_dong.fillna(0)

    gn_ppl_dong = (df_geo[df_geo['is_early'] & (df_geo['후보자'] == '선거인수')].groupby(dong_key)['득표수'].sum().reset_index(name='_gn_ppl'))
    df_dong = df_dong.merge(gn_ppl_dong, on=dong_key, how='left')
    df_dong['sum_people'] = df_dong['sum_people'] + df_dong['_gn_ppl'].fillna(0)
    df_dong.drop(columns=['_gn_ppl'], inplace=True)

    df_dong['sgg_candidates'] = df_dong['선거구명'].apply(sgg_cands)
    df_dong['primary_sgg'] = df_dong['sgg_candidates'].apply(lambda x: x[0] if x else "")
    df_dong['dong_norm'] = df_dong['법정읍면동명'].apply(normalize_dong_name)
    df_dong['province_tag'] = df_dong['시도명'].map(PROV_FULL_TO_SHORT).fillna(df_dong['시도명'])
    df_dong['area2_name'] = df_dong['선거구명']

    mask_seoul = df_dong['province_tag'] == 'Seoul'
    mask_g3 = df_dong['primary_sgg'].isin(['강남', '서초', '송파'])
    df_dong.loc[mask_seoul & mask_g3, 'province_tag'] = 'Seoul (Gangnam3gu)'
    df_dong.loc[mask_seoul & ~mask_g3, 'province_tag'] = 'Seoul (Non-Gangnam3gu)'

    df_gw = df[df['법정읍면동명'].isin(special_dong_names)]
    df_gw_v = df_gw[~df_gw['is_meta']]

    go_dem_c = df_gw_v[df_gw_v['is_dem']].groupby(const_key)['득표수'].sum().reset_index(name='out_precinct_early_dem')
    go_con_c = df_gw_v[df_gw_v['is_con']].groupby(const_key)['득표수'].sum().reset_index(name='out_precinct_early_con')
    go_tot_c = df_gw_v.groupby(const_key)['득표수'].sum().reset_index(name='out_precinct_early_total')
    go_turn_c = (df_gw[df_gw['후보자'] == '투표수'].groupby(const_key)['득표수'].sum().reset_index(name='out_precinct_early_turnout'))

    df_const = df_dong.groupby(const_key)['sum_people'].sum().reset_index(name='sum_people')
    for frame in (go_dem_c, go_con_c, go_tot_c, go_turn_c):
        df_const = df_const.merge(frame, on=const_key, how='left')
    df_const = df_const.fillna(0)

    return df_dong, df_const, pd.DataFrame()


# ============================================================================
# Optional coarser province grouping: 4 megaregions instead of ~17-18
# individual provinces/metro-cities.
# ============================================================================
MEGAREGION_MAP = {
    'Seoul': '수도권', 'Gyeonggi': '수도권', 'Incheon': '수도권',
    'Seoul (Gangnam3gu)': '수도권', 'Seoul (Non-Gangnam3gu)': '수도권',
    'Gangwon': '강원충청권', 'Chungbuk': '강원충청권', 'Chungnam': '강원충청권',
    'Daejeon': '강원충청권', 'Sejong': '강원충청권',
    'Busan': '영남권', 'Daegu': '영남권', 'Ulsan': '영남권',
    'Gyeongbuk': '영남권', 'Gyeongnam': '영남권',
    'Gwangju': '제주호남권', 'Jeonbuk': '제주호남권', 'Jeonnam': '제주호남권', 'Jeju': '제주호남권',
    '서울': '수도권', '경기': '수도권', '인천': '수도권',
    '강원': '강원충청권', '충북': '강원충청권', '충남': '강원충청권',
    '대전': '강원충청권', '세종': '강원충청권',
    '부산': '영남권', '대구': '영남권', '울산': '영남권', '경북': '영남권', '경남': '영남권',
    '광주': '제주호남권', '전북': '제주호남권', '전남': '제주호남권', '제주': '제주호남권',
    '서울특별시': '수도권', '경기도': '수도권', '인천광역시': '수도권',
    '강원도': '강원충청권', '강원특별자치도': '강원충청권', '충청북도': '강원충청권', '충청남도': '강원충청권',
    '대전광역시': '강원충청권', '세종특별자치시': '강원충청권',
    '부산광역시': '영남권', '대구광역시': '영남권', '울산광역시': '영남권',
    '경상북도': '영남권', '경상남도': '영남권',
    '광주광역시': '제주호남권', '전라북도': '제주호남권', '전북특별자치도': '제주호남권',
    '전라남도': '제주호남권', '제주특별자치도': '제주호남권',
}


def megaregion_of(province_tag_series):
    mapped = province_tag_series.map(MEGAREGION_MAP)
    unmapped = province_tag_series[mapped.isna()].unique()
    if len(unmapped) > 0:
        raise ValueError(f"megaregion_of: no megaregion mapping for {sorted(unmapped.tolist())} "
                          f"-- add these to MEGAREGION_MAP before proceeding. "
                          f"Known keys: {sorted(MEGAREGION_MAP.keys())}")
    return mapped


# PROVINCE_GROUPING: 'original' (~17-18 individual provinces, DEFAULT) or 'megaregion' (4
# regions). Settable via MEBANE_PROVINCE_GROUPING.
PROVINCE_GROUPING = os.environ.get('MEBANE_PROVINCE_GROUPING', 'original')
if PROVINCE_GROUPING not in ('megaregion', 'original'):
    raise ValueError(f"MEBANE_PROVINCE_GROUPING must be 'megaregion' or 'original', got '{PROVINCE_GROUPING}'")
print(f"[mebane_r_pipeline] Province grouping mode: '{PROVINCE_GROUPING}' "
      f"(set MEBANE_PROVINCE_GROUPING=megaregion|original to change; unset = original).")


def _valid_election_ids():
    return sorted(str(key) for key in ELECTION_CONFIGS)


def _resolve_election_key(elec_id):
    if elec_id in GENERAL_ELECTION_INT_KEYS:
        elec_id = int(elec_id)
    return elec_id if elec_id in ELECTION_CONFIGS else None


def compute_prov_code(province_tag_series, grouping=None):
    grouping = grouping if grouping is not None else PROVINCE_GROUPING
    if grouping == 'megaregion':
        group_label = megaregion_of(province_tag_series)
    elif grouping == 'original':
        group_label = province_tag_series
    else:
        raise ValueError(f"compute_prov_code: grouping must be 'megaregion' or 'original', got '{grouping}'")
    return group_label.astype('category').cat.codes, group_label


# ============================================================================
# Unit-level table construction, filtering, and null-data generators
# ============================================================================

def _attach_out_precinct_share(dm, df_const):
    const_key = ['시도명', '선거구명']
    if df_const.empty or not set(const_key).issubset(dm.columns):
        dm['out_precinct_share'] = np.nan
        return dm
    in_sum = dm.groupby(const_key)['in_precinct_early_total'].sum().reset_index(name='_in_precinct_const_sum')
    share = df_const[const_key + ['out_precinct_early_total']].merge(in_sum, on=const_key, how='left').fillna(0)
    denom = share['out_precinct_early_total'] + share['_in_precinct_const_sum']
    share['out_precinct_share'] = np.where(denom > 0, share['out_precinct_early_total'] / denom, np.nan)
    return dm.merge(share[const_key + ['out_precinct_share']], on=const_key, how='left')


def load_pure_unit_table(elec_id, ch, leader_side='dem'):
    if leader_side not in ('dem', 'con'):
        raise ValueError(f"leader_side must be 'dem' or 'con', got '{leader_side}'")
    elec_key = _resolve_election_key(elec_id)
    if elec_key is None:
        raise ValueError(f"Unrecognized election id: '{elec_id}'. Valid ids: {_valid_election_ids()}. "
                          f"These are internal short codes, not Korean administrative region codes.")
    cfg = ELECTION_CONFIGS[elec_key]

    df_dong, df_const, _ = load_election_csv(
        cfg['result_csv'], dem_pattern=cfg['dem_pattern'], con_pattern=cfg['con_pattern'],
        third_pattern=cfg.get('third_pattern'), election_type=cfg['election_type'])

    if df_dong.empty:
        return pd.DataFrame()

    is_no_early = bool(cfg.get('no_early_voting'))
    dm = df_dong.copy()
    if is_no_early:
        dm = dm[dm['same_day_total'] > MIN_VOTES].copy()
    else:
        dm = dm[(dm['in_precinct_early_total'] > MIN_VOTES) & (dm['same_day_total'] > MIN_VOTES)].copy()
    # This unit-drop filter (zero leader votes in both channels) always uses the dem
    # columns regardless of leader_side, so that leader_side='dem' and leader_side='con'
    # runs are fit on the IDENTICAL unit set -- this isolates the effect of *who is coded
    # as leader* from any effect of *which units pass the filter*, which is what a clean
    # bidirectional comparison requires. Units with zero conservative votes in both
    # channels essentially never occur in these elections anyway.
    dm = dm[~((dm['in_precinct_early_dem'] == 0) & (dm['same_day_dem'] == 0))].copy()
    dm = _attach_out_precinct_share(dm, df_const)
    dm['pooled_dem'] = dm['in_precinct_early_dem'] + dm['same_day_dem']
    dm['pooled_con'] = dm['in_precinct_early_con'] + dm['same_day_con']
    dm['pooled_total'] = dm['in_precinct_early_total'] + dm['same_day_total']

    suffix = leader_side  # 'dem' or 'con'
    mapping = {
        'early': (f'in_precinct_early_{suffix}', 'in_precinct_early_total'),
        'sameday': (f'same_day_{suffix}', 'same_day_total'),
        'pooled': (f'pooled_{suffix}', 'pooled_total'),
        'total_no_early': (f'same_day_{suffix}', 'same_day_total'),
    }
    if ch not in mapping:
        raise ValueError(f"Unknown channel '{ch}' for level='dong'. Valid channels: {sorted(mapping)}")

    leader_c, tot_c = mapping[ch]
    out = dm[['province_tag', '시도명', '선거구명', '법정읍면동명', 'out_precinct_share', leader_c, tot_c, 'sum_people']].rename(
        columns={leader_c: 'leader_votes', tot_c: 'total_votes', 'sum_people': 'eligible', '선거구명': 'constituency', '법정읍면동명': 'dong_name'})

    out['eligible'] = out['eligible'].clip(lower=out['total_votes'])
    out['tau'] = out['total_votes'] / out['eligible']
    out['nu'] = out['leader_votes'] / out['total_votes']
    out = out[(out['tau'] > 0.001) & (out['tau'] < 0.999) & (out['nu'] > 0.001) & (out['nu'] < 0.999)].copy()
    out['prov_code'], out['prov_group'] = compute_prov_code(out['province_tag'])
    return out


def load_pure_constituency_table(elec_id, ch, leader_side='dem'):
    if leader_side not in ('dem', 'con'):
        raise ValueError(f"leader_side must be 'dem' or 'con', got '{leader_side}'")
    elec_key = _resolve_election_key(elec_id)
    if elec_key is None:
        raise ValueError(f"Unrecognized election id: '{elec_id}'. Valid ids: {_valid_election_ids()}. "
                          f"These are internal short codes, not Korean administrative region codes.")
    cfg = ELECTION_CONFIGS[elec_key]

    df_dong, df_const, _ = load_election_csv(
        cfg['result_csv'], dem_pattern=cfg['dem_pattern'], con_pattern=cfg['con_pattern'],
        third_pattern=cfg.get('third_pattern'), election_type=cfg['election_type'])

    if df_dong.empty:
        return pd.DataFrame()

    group_cols = ['province_tag', '시도명', '선거구명']
    sum_cols = ['in_precinct_early_dem', 'in_precinct_early_con', 'in_precinct_early_total',
                'same_day_dem', 'same_day_con', 'same_day_total', 'sum_people']
    dm = df_dong.groupby(group_cols, as_index=False)[sum_cols].sum()

    const_cols = ['out_precinct_early_dem', 'out_precinct_early_con', 'out_precinct_early_total']
    if not df_const.empty and set(['시도명', '선거구명']).issubset(df_const.columns):
        dm = dm.merge(df_const[['시도명', '선거구명'] + const_cols], on=['시도명', '선거구명'], how='left')
    else:
        for c in const_cols:
            dm[c] = 0.0
    for c in const_cols:
        dm[c] = dm[c].fillna(0)

    # See note in load_pure_unit_table: the unit-drop filter stays dem-based regardless of
    # leader_side, so leader_side='dem' and 'con' runs share the identical unit set.
    is_no_early = bool(cfg.get('no_early_voting'))
    if is_no_early:
        dm = dm[dm['same_day_total'] > MIN_VOTES].copy()
    else:
        dm = dm[(dm['in_precinct_early_total'] > MIN_VOTES) & (dm['same_day_total'] > MIN_VOTES)].copy()
    dm = dm[~((dm['in_precinct_early_dem'] == 0) & (dm['same_day_dem'] == 0))].copy()
    dm = _attach_out_precinct_share(dm, df_const)

    dm['early_total_dem'] = dm['in_precinct_early_dem'] + dm['out_precinct_early_dem']
    dm['early_total_con'] = dm['in_precinct_early_con'] + dm['out_precinct_early_con']
    dm['early_total_total'] = dm['in_precinct_early_total'] + dm['out_precinct_early_total']
    dm['total_dem'] = dm['early_total_dem'] + dm['same_day_dem']
    dm['total_con'] = dm['early_total_con'] + dm['same_day_con']
    dm['total_total'] = dm['early_total_total'] + dm['same_day_total']

    suffix = leader_side  # 'dem' or 'con'
    mapping = {
        'early': (f'in_precinct_early_{suffix}', 'in_precinct_early_total'),
        'early_out': (f'out_precinct_early_{suffix}', 'out_precinct_early_total'),
        'early_total': (f'early_total_{suffix}', 'early_total_total'),
        'sameday': (f'same_day_{suffix}', 'same_day_total'),
        'total': (f'total_{suffix}', 'total_total'),
        'total_no_early': (f'same_day_{suffix}', 'same_day_total'),
    }
    if ch not in mapping:
        raise ValueError(f"Unknown channel '{ch}' for level='constituency'. Valid channels: {sorted(mapping)}")

    leader_c, tot_c = mapping[ch]
    out = dm[['province_tag', '시도명', '선거구명', 'out_precinct_share', leader_c, tot_c, 'sum_people']].rename(
        columns={leader_c: 'leader_votes', tot_c: 'total_votes', 'sum_people': 'eligible', '선거구명': 'constituency'})

    out['eligible'] = out['eligible'].clip(lower=out['total_votes'])
    out['tau'] = out['total_votes'] / out['eligible']
    out['nu'] = out['leader_votes'] / out['total_votes']
    out = out[(out['tau'] > 0.001) & (out['tau'] < 0.999) & (out['nu'] > 0.001) & (out['nu'] < 0.999)].copy()
    out['prov_code'], out['prov_group'] = compute_prov_code(out['province_tag'])
    return out


def generate_marginal_null(real_sub, seed=1):
    rng = np.random.default_rng(seed)

    def beta_params(p, k=200):
        return np.clip(p * k, 0.5, None), np.clip((1 - p) * k, 0.5, None)

    a_tau, b_tau = beta_params(real_sub['tau'].values)
    a_nu, b_nu = beta_params(real_sub['nu'].values)
    sim = real_sub.copy()
    sim['tau'] = np.clip(rng.beta(a_tau, b_tau), 0.001, 0.999)
    sim['nu'] = np.clip(rng.beta(a_nu, b_nu), 0.001, 0.999)
    sim['total_votes'] = np.round(sim['eligible'] * sim['tau']).clip(lower=1)
    sim['leader_votes'] = np.round(sim['total_votes'] * sim['nu'])
    return sim


def generate_joint_null(real_sub, seed=1):
    prov_code = real_sub['prov_code'].values
    n_prov = int(prov_code.max()) + 1

    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    y_real = np.stack([logit(real_sub['tau'].values), logit(real_sub['nu'].values)], axis=-1)
    prov_means = np.zeros((n_prov, 2))
    for p in range(n_prov):
        mask = prov_code == p
        prov_means[p] = y_real[mask].mean(axis=0) if mask.sum() > 0 else y_real.mean(axis=0)

    resid = y_real - prov_means[prov_code]
    cov = np.cov(resid.T) + np.eye(2) * 1e-6
    rng = np.random.default_rng(seed)
    y_sim = prov_means[prov_code] + rng.multivariate_normal([0., 0.], cov, size=len(real_sub))

    sim = real_sub.copy()
    sim['tau'] = np.clip(1 / (1 + np.exp(-y_sim[:, 0])), 0.001, 0.999)
    sim['nu'] = np.clip(1 / (1 + np.exp(-y_sim[:, 1])), 0.001, 0.999)
    sim['total_votes'] = np.round(sim['eligible'] * sim['tau']).clip(lower=1)
    sim['leader_votes'] = np.round(sim['total_votes'] * sim['nu'])
    return sim


# ============================================================================
# R/JAGS invocation
# ============================================================================

def run_r_model(df, r_n_iter=3000, r_n_chains=4, r_burn_in=1000, use_parcomp=True, r_n_adapt=1000, work_dir=None):
    if shutil.which('Rscript') is None:
        raise FileNotFoundError("run_r_model: 'Rscript' not found on PATH.")

    work_dir = work_dir or tempfile.mkdtemp(prefix='mebane_r_pipeline_')
    input_path = os.path.join(work_dir, 'input.csv')
    output_path = os.path.join(work_dir, 'output.csv')
    units_path = os.path.join(work_dir, 'output_units.csv')

    required = ['leader_votes', 'total_votes', 'eligible', 'prov_code']
    df[required].to_csv(input_path, index=False)

    argv = ['Rscript', R_SCRIPT_PATH, input_path, output_path,
            str(r_n_iter), str(r_n_chains), str(r_burn_in),
            'TRUE' if use_parcomp else 'FALSE', str(r_n_adapt)]

    print(f"[run_r_model] MCMC settings: n_iter={r_n_iter}, n_chains={r_n_chains}, "
          f"burn_in={r_burn_in}, n_adapt={r_n_adapt}, parComp={use_parcomp}")
    print(f"[run_r_model] {' '.join(argv)}")
    t0 = time.time()
    result = subprocess.run(argv, capture_output=True, text=True)
    elapsed = time.time() - t0

    print(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(f"R script crashed: {result.stderr}")

    return pd.read_csv(output_path).iloc[0].to_dict(), pd.read_csv(units_path), elapsed


# ============================================================================
# Single-cell driver
# ============================================================================

def fit_one_cell(elec_id, level, channel, source, leader_side='dem', r_n_iter=3000, r_n_chains=4,
                  r_burn_in=1000, use_parcomp=True, r_n_adapt=1000):
    if leader_side not in ('dem', 'con'):
        raise ValueError(f"leader_side must be 'dem' or 'con', got '{leader_side}'")
    if level == 'dong':
        df = load_pure_unit_table(elec_id, channel, leader_side=leader_side)
    else:
        df = load_pure_constituency_table(elec_id, channel, leader_side=leader_side)

    if df.empty:
        raise ValueError(f"No data matching criteria: {elec_id}/{level}/{channel}/{leader_side}")

    if source == 'marginal_null':
        df = generate_marginal_null(df)
    elif source == 'joint_null':
        df = generate_joint_null(df)
    elif source != 'real':
        raise ValueError(f"Unknown source '{source}', expected real/marginal_null/joint_null")

    cell_tag = f"{elec_id}/{level}/{channel}/{source}/leader={leader_side}"
    print(f"\n{'='*70}\nCELL: {cell_tag}  N={len(df)}\n{'='*70}")
    summary, units, elapsed = run_r_model(df, r_n_iter=r_n_iter, r_n_chains=r_n_chains,
                                           r_burn_in=r_burn_in, use_parcomp=use_parcomp,
                                           r_n_adapt=r_n_adapt)

    def g(key, default=None):
        v = summary.get(key, default)
        if isinstance(v, float) and np.isnan(v):
            return default
        return v

    print(f"\n{'-'*70}\nRESULT: {cell_tag}\n{'-'*70}")

    def fmt_ci(median_key, lo_key, hi_key):
        med, lo, hi = g(median_key), g(lo_key), g(hi_key)
        if med is None:
            return "N/A"
        if lo is None or hi is None:
            return f"{med:.4f}"
        return f"{med:.4f} [{lo:.4f}, {hi:.4f}]"

    print(f"p_no_fraud_median      : {fmt_ci('p_no_fraud_median', 'p_no_fraud_hpd95_lo', 'p_no_fraud_hpd95_hi')}")
    print(f"p_incremental_median   : {fmt_ci('p_incremental_median', 'p_incremental_hpd95_lo', 'p_incremental_hpd95_hi')}")
    print(f"p_extreme_median       : {fmt_ci('p_extreme_median', 'p_extreme_hpd95_lo', 'p_extreme_hpd95_hi')}")
    p_incr, p_ext = g('p_incremental_median'), g('p_extreme_median')
    if p_incr is not None and p_ext is not None:
        print(f"fraud_share_median     : {p_incr + p_ext:.4f}")

    def fmt_fw(mean_key, lo_key, hi_key):
        mean, lo, hi = g(mean_key), g(lo_key), g(hi_key)
        if mean is None:
            return "N/A"
        if lo is None or hi is None:
            return f"{mean:.1f}"
        return f"{mean:.1f} [{lo:.1f}, {hi:.1f}] (conservative 95% interval)"

    print(f"Ft_total (manufactured): {fmt_fw('Ft_total', 'Ft_total_hpd95_lo', 'Ft_total_hpd95_hi')}")
    print(f"Fw_total (total fraud) : {fmt_fw('Fw_total', 'Fw_total_hpd95_lo', 'Fw_total_hpd95_hi')}")

    n_flagged, total_units = g('n_regions_flagged'), g('total_units')
    if n_flagged is not None and total_units is not None:
        print(f"n_regions_flagged      : {n_flagged:.0f}/{total_units:.0f}")
    rhat = g('rhat_pi')
    if rhat is not None:
        print(f"rhat_pi                : {rhat:.4f}" + ("  *** NOT CONVERGED (>1.05) ***" if rhat > 1.05 else ""))

    M_vals = [g('M_pi_no_fraud'), g('M_pi_incremental'), g('M_pi_extreme')]
    D_vals = [g('D_pi_no_fraud'), g('D_pi_incremental'), g('D_pi_extreme')]
    if all(v is not None for v in M_vals):
        print(f"M(pi_j) [chain-mean gap]: {M_vals[0]:.4f}/{M_vals[1]:.4f}/{M_vals[2]:.4f}")
    if all(v is not None for v in D_vals):
        print(f"D(pi_j) [dip test p]   : {D_vals[0]:.4f}/{D_vals[1]:.4f}/{D_vals[2]:.4f}")
        flagged = [name for name, m, d in zip(['pi1', 'pi2', 'pi3'], M_vals, D_vals)
                   if m is not None and m > 0.01 and (d is None or d < 0.05)]
        if flagged:
            print(f"*** Mebane (2023) flag: {', '.join(flagged)} show non-trivial cross-chain mean "
                  f"difference alongside a small dip-test p-value -- signal of lost votes / model "
                  f"misspecification, not just an MCMC convergence problem. ***")

    print(f"n_adapt                : {r_n_adapt}")
    print(f"elapsed_seconds        : {elapsed:.1f}")
    print(f"\nEND: {cell_tag}")
    return summary, units


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: mebane_r_pipeline_bidirectional.py <elec_id> <level> <channel> <source> "
              "[leader_side] [r_n_iter] [r_n_chains] [r_burn_in] [use_parcomp] [r_n_adapt]")
        print("  leader_side: 'dem' (default) or 'con'")
        sys.exit(1)

    elec_id = sys.argv[1]
    level = sys.argv[2]
    channel = sys.argv[3]
    source = sys.argv[4]
    leader_side = sys.argv[5] if len(sys.argv) > 5 else 'dem'
    r_n_iter = int(sys.argv[6]) if len(sys.argv) > 6 else 3000
    r_n_chains = int(sys.argv[7]) if len(sys.argv) > 7 else 4
    r_burn_in = int(sys.argv[8]) if len(sys.argv) > 8 else 1000
    use_parcomp = bool(int(sys.argv[9])) if len(sys.argv) > 9 else True
    r_n_adapt = int(sys.argv[10]) if len(sys.argv) > 10 else 1000

    fit_one_cell(elec_id, level, channel, source, leader_side=leader_side, r_n_iter=r_n_iter,
                 r_n_chains=r_n_chains, r_burn_in=r_burn_in, use_parcomp=use_parcomp,
                 r_n_adapt=r_n_adapt)

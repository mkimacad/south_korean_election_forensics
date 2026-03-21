import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import re
import glob
from difflib import get_close_matches
from scipy.stats import chisquare, pearsonr, norm, probplot, normaltest
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ==========================================
# KOREAN FONT SETUP
# ==========================================

def setup_korean_font():
    """Configure matplotlib to render Korean (Hangul) text."""
    CANDIDATES = [
        ('Noto Sans CJK', None), ('Noto Serif CJK', None),
        ('NanumSquare', None), ('NanumSquare_ac', None),
        ('NanumGothic', None), ('NanumBarunGothic', None),
        ('Malgun Gothic', None), ('Apple SD Gothic Neo', None),
        ('UnDotum', None), ('Baekmuk', None),
    ]

    all_names = {f.name for f in fm.fontManager.ttflist}
    for substr, _ in CANDIDATES:
        match = next((n for n in all_names if substr in n), None)
        if match:
            matplotlib.rcParams['font.family'] = match
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"    Korean font set to: '{match}'")
            return

    noto_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
        '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc',
    ]
    for path in noto_paths:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            matplotlib.rcParams['font.family'] = prop.get_name()
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"    Korean font loaded from: {path}")
            return

    print("    No Korean font found – attempting: apt-get install fonts-nanum ...")
    import subprocess, shutil
    if shutil.which('apt-get'):
        try:
            subprocess.run(['apt-get', 'install', '-y', '-q', 'fonts-nanum-extra'],
                           check=True, capture_output=True)
            fm.fontManager.__init__()
            all_names = {f.name for f in fm.fontManager.ttflist}
            match = next((n for n in all_names if 'NanumSquare' in n), None)
            if match:
                matplotlib.rcParams['font.family'] = match
                matplotlib.rcParams['axes.unicode_minus'] = False
                print(f"    Korean font installed and set to: '{match}'")
                return
        except Exception as e:
            print(f"    apt-get failed: {e}")

# ==========================================
# CONFIGURATION
# ==========================================
ELECTION_NUM = 21

ELECTION_CONFIGS = {
    21: {
        'census_csv':    '21st_election_census.csv',
        'result_csv':    '21st_election_result.csv',
        'apt_csv_glob':  '*21st_election_*_apt_price.csv', 
        'dem_pattern':   r'더불어민주당',
        'con_pattern':   r'미래통합당|자유한국당',
        'label':         '21st General Election (2020)',
        'dashboard_out': 'mega_forensics_dashboard_21st.png',
    },
    22: {
        'census_csv':    '22nd_election_census.csv',
        'result_csv':    '22nd_election_result.csv',
        'apt_csv_glob':  '*22nd_election_*_apt_price.csv', 
        'dem_pattern':   r'더불어민주당',
        'con_pattern':   r'국민의힘',
        'label':         '22nd General Election (2024)',
        'dashboard_out': 'mega_forensics_dashboard_22nd.png',
    },
}

CFG = ELECTION_CONFIGS[ELECTION_NUM]

SPECIAL_DONG_NAMES = {
    '거소·선상투표', '관외사전투표', '국외부재자투표',
    '국외부재자투표(공관)', '잘못 투입·구분된 투표지',
}

GWANNAESA_LABEL = '관내사전투표'
GWANOE_LABEL    = '관외사전투표'
META_CANDIDATES = {'선거인수', '투표수', '무효 투표수', '기권자수'}

PROV_FULL_TO_SHORT = {
    '서울특별시': '서울', '부산광역시': '부산', '대구광역시': '대구',
    '인천광역시': '인천', '광주광역시': '광주', '대전광역시': '대전',
    '울산광역시': '울산', '세종특별자치시': '세종',
    '경기도': '경기', '강원도': '강원', '강원특별자치도': '강원',
    '충청북도': '충북', '충청남도': '충남',
    '전라북도': '전북', '전북특별자치도': '전북', '전라남도': '전남',
    '경상북도': '경북', '경상남도': '경남', '제주특별자치도': '제주',
}

# ==========================================
# SHARED NAME NORMALISATION UTILITIES
# ==========================================

def normalize_dong_name(name: str) -> str:
    if not isinstance(name, str): return ""
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
    if buf: tokens.append(''.join(buf))
    return [t for t in tokens if t]

def normalize_sigungu(name: str) -> list:
    if not isinstance(name, str): return []
    name = re.sub(r'\(.*?\)', '', name).strip()
    if not name: return []
    tokens = split_admin_tokens(name)
    if not tokens:
        stripped = re.sub(r'[시군구갑을병정무]$', '', name).strip()
        return [stripped] if stripped else []
    si_gun_count = sum(1 for t in tokens if t[-1] in '시군' and len(t) >= 2)
    gu_count     = sum(1 for t in tokens if t[-1] == '구' and len(t) >= 2)
    ordered = tokens if (si_gun_count >= 2 or (si_gun_count == 0 and gu_count >= 2)) else list(reversed(tokens))
    candidates = []
    for t in ordered:
        key = re.sub(r'[시군구]$', '', t).strip()
        if key and key not in candidates: candidates.append(key)
    return candidates

def get_urban_type(name: str) -> str:
    if pd.isna(name): return 'Unknown'
    name = str(name).strip()
    if re.search(r'(읍)\d*$', name): return 'Eup'
    elif re.search(r'(면)\d*$', name): return 'Myeon'
    else: return 'Dong'

def check_military_zone(row):
    sgg = str(row.get('area2_name', ''))
    dong = str(row.get('name', ''))
    military_dongs = [
        '진동면', '군내면', '장단면', '파평면', '중면', '장남면', '백학면', '왕징면', 
        '근북면', '근동면', '원동면', '원남면', '임남면', '동송읍', '철원읍', 
        '상서면', '서화면', '방산면', '해안면', '현내면', 
        '백령면', '대청면', '연평면', '신도안면', '오천읍', '고경면'
    ]
    if any(m_dong in dong for m_dong in military_dongs): return 1
    if '진해구' in sgg: return 1
    return 0

# ==========================================
# 1. DEMOGRAPHIC & ASSET DATA LOADERS
# ==========================================

def _detect_year_prefix(df: pd.DataFrame) -> str:
    for col in df.columns:
        m = re.match(r'(\d{4}년\d{2}월)_계_총인구수', col)
        if m: return m.group(1)
    raise ValueError("Cannot detect census year prefix.")

def load_census_csv(csv_path: str) -> pd.DataFrame:
    print(f"\n--- [1/5] Loading Demographic Census Data (5-year intervals) ---")
    if not os.path.exists(csv_path): return pd.DataFrame()
    try:
        try: df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError: df = pd.read_csv(csv_path, encoding='cp949', low_memory=False)

        prefix = _detect_year_prefix(df)
        
        cols_1824 = [f"{prefix}_계_{a}세" for a in range(18, 25)]
        cols_2529 = [f"{prefix}_계_{a}세" for a in range(25, 30)]
        cols_3034 = [f"{prefix}_계_{a}세" for a in range(30, 35)]
        cols_3539 = [f"{prefix}_계_{a}세" for a in range(35, 40)]
        cols_4044 = [f"{prefix}_계_{a}세" for a in range(40, 45)]
        cols_4549 = [f"{prefix}_계_{a}세" for a in range(45, 50)]
        cols_5054 = [f"{prefix}_계_{a}세" for a in range(50, 55)]
        cols_5559 = [f"{prefix}_계_{a}세" for a in range(55, 60)]
        cols_6064 = [f"{prefix}_계_{a}세" for a in range(60, 65)]
        cols_6569 = [f"{prefix}_계_{a}세" for a in range(65, 70)]
        cols_70plus = [f"{prefix}_계_{a}세" for a in range(70, 100)] + [f"{prefix}_계_100세 이상"]
        
        cols_male_all = [f"{prefix}_남_{a}세" for a in range(18, 100)] + [f"{prefix}_남_100세 이상"]
        voting_age_cols = ([f"{prefix}_계_{a}세" for a in range(18, 100)] + [f"{prefix}_계_100세 이상"])
        cols_4059 = [f"{prefix}_계_{a}세" for a in range(40, 60)]

        all_target_cols = cols_1824 + cols_2529 + cols_3034 + cols_3539 + cols_4044 + cols_4549 + \
                          cols_5054 + cols_5559 + cols_6064 + cols_6569 + cols_70plus + cols_male_all + voting_age_cols
        
        for col in set(all_target_cols):
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False).pipe(pd.to_numeric, errors='coerce').fillna(0)

        df = df.copy() 
        df['total_voting_pop'] = df[[c for c in voting_age_cols if c in df.columns]].sum(axis=1)
        df = df[df['total_voting_pop'] > 0].copy()

        df['pct_1824']   = df[[c for c in cols_1824 if c in df.columns]].sum(axis=1) / df['total_voting_pop']
        df['pct_2529']   = df[[c for c in cols_2529 if c in df.columns]].sum(axis=1) / df['total_voting_pop']
        df['pct_3034']   = df[[c for c in cols_3034 if c in df.columns]].sum(axis=1) / df['total_voting_pop']
        df['pct_3539']   = df[[c for c in cols_3539 if c in df.columns]].sum(axis=1) / df['total_voting_pop']
        df['pct_4044']   = df[[c for c in cols_4044 if c in df.columns]].sum(axis=1) / df['total_voting_pop']
        df['pct_4549']   = df[[c for c in cols_4549 if c in df.columns]].sum(axis=1) / df['total_voting_pop']
        df['pct_5054']   = df[[c for c in cols_5054 if c in df.columns]].sum(axis=1) / df['total_voting_pop']
        df['pct_5559']   = df[[c for c in cols_5559 if c in df.columns]].sum(axis=1) / df['total_voting_pop']
        df['pct_6064']   = df[[c for c in cols_6064 if c in df.columns]].sum(axis=1) / df['total_voting_pop']
        df['pct_6569']   = df[[c for c in cols_6569 if c in df.columns]].sum(axis=1) / df['total_voting_pop']
        df['pct_70plus'] = df[[c for c in cols_70plus if c in df.columns]].sum(axis=1) / df['total_voting_pop']

        df['pct_male']               = df[[c for c in cols_male_all if c in df.columns]].sum(axis=1) / df['total_voting_pop']
        df['demographic_propensity'] = df[[c for c in cols_4059 if c in df.columns]].sum(axis=1) / df['total_voting_pop']

        def extract_census_keys(admin_name):
            if not isinstance(admin_name, str): return [], ""
            clean = re.sub(r'\(.*?\)', '', admin_name).strip()
            parts = clean.split()
            dong_norm = normalize_dong_name(parts[-1]) if parts else ""
            sgg_cands = []
            for token in reversed(parts[:-1]):
                for c in normalize_sigungu(token):
                    if c not in sgg_cands: sgg_cands.append(c)
            return sgg_cands, dong_norm

        rows = []
        for _, row in df.iterrows():
            sgg_cands, dong_norm = extract_census_keys(row['행정구역'])
            rows.append({
                'sgg_candidates': sgg_cands, 'primary_sgg': sgg_cands[0] if sgg_cands else "",
                'dong_norm': dong_norm, 'dong_raw': row['행정구역'],
                'demographic_propensity': row['demographic_propensity'],
                'pct_1824': row['pct_1824'], 'pct_2529': row['pct_2529'],
                'pct_3034': row['pct_3034'], 'pct_3539': row['pct_3539'],
                'pct_4044': row['pct_4044'], 'pct_4549': row['pct_4549'],
                'pct_5054': row['pct_5054'], 'pct_5559': row['pct_5559'],
                'pct_6064': row['pct_6064'], 'pct_6569': row['pct_6569'],
                'pct_70plus': row['pct_70plus'], 'pct_male': row['pct_male'],
            })

        census = pd.DataFrame(rows)
        print(f"    Loaded {len(census):,} census rows with 5-year intervals.")
        return census
    except Exception as e:
        print(f"[!] Error processing census CSV: {e}")
        return pd.DataFrame()

def load_apt_csv(glob_pattern: str) -> pd.DataFrame:
    print(f"\n--- [2/5] Loading Apartment Transaction Data ({glob_pattern}) ---")
    file_list = glob.glob(glob_pattern)
    
    if not file_list:
        print("[!] No APT CSVs found matching the pattern. Wealth proxy will be skipped.")
        return pd.DataFrame()
        
    df_list = []
    for file in file_list:
        try:
            try: df_temp = pd.read_csv(file, encoding='utf-8', skiprows=15)
            except UnicodeDecodeError: df_temp = pd.read_csv(file, encoding='cp949', skiprows=15)
            df_list.append(df_temp)
        except Exception as e:
            print(f"    [!] Error reading {file}: {e}")
            
    if not df_list:
        return pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)
    
    try:
        df['거래금액(만원)'] = pd.to_numeric(df['거래금액(만원)'].astype(str).str.replace(',', '').str.strip(), errors='coerce')
        df['전용면적(㎡)'] = pd.to_numeric(df['전용면적(㎡)'], errors='coerce')
        df['price_per_sqm'] = df['거래금액(만원)'] / df['전용면적(㎡)']
        
        def parse_loc(x):
            parts = str(x).split()
            prov = PROV_FULL_TO_SHORT.get(parts[0], parts[0]) if len(parts) > 0 else ""
            if len(parts) > 2:
                sgg_list = normalize_sigungu(parts[1])
                sgg = sgg_list[0] if sgg_list else ""
            else:
                sgg = "" 
            dong = normalize_dong_name(parts[-1]) if len(parts) > 0 else ""
            return pd.Series([prov, sgg, dong])
            
        df[['prov', 'sgg', 'dong_norm']] = df['시군구'].apply(parse_loc)
        
        apt_agg = df.groupby(['prov', 'sgg', 'dong_norm'])['price_per_sqm'].median().reset_index()
        apt_agg.rename(columns={'price_per_sqm': 'median_apt_price_sqm'}, inplace=True)
        print(f"    Calculated stable median prices for {len(apt_agg):,} unique Dongs.")
        return apt_agg
    except Exception as e:
        print(f"[!] Error processing concatenated APT data: {e}")
        return pd.DataFrame()


# ==========================================
# 2. ELECTION CSV LOADER
# ==========================================

def load_election_csv(csv_path: str, dem_pattern: str, con_pattern: str):
    print(f"\n--- [3/5] Loading Election Result Data ({csv_path}) ---")
    try:
        try: df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError: df = pd.read_csv(csv_path, encoding='cp949', low_memory=False)
    except Exception as e:
        print(f"[!] Failed to read election CSV: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df['득표수']       = pd.to_numeric(df['득표수'], errors='coerce').fillna(0).astype(int)
    df['is_dem']      = df['후보자'].str.contains(dem_pattern, case=False, na=False)
    df['is_con']      = df['후보자'].str.contains(con_pattern, case=False, na=False)
    df['is_meta']     = df['후보자'].isin(META_CANDIDATES)
    df['is_gwannaesa']= df['투표구명'] == GWANNAESA_LABEL

    dong_key  = ['시도명', '선거구명', '법정읍면동명']
    const_key = ['시도명', '선거구명']

    def sgg_cands_from_constituency(name):
        if not isinstance(name, str): return []
        if '_' in name: return normalize_sigungu(name.split('_', 1)[1])
        return normalize_sigungu(re.sub(r'[갑을병정무]$', '', name).strip())

    df_geo   = df[~df['법정읍면동명'].isin(SPECIAL_DONG_NAMES)].copy()
    df_votes = df_geo[~df_geo['is_meta']].copy()

    gn_dem = df_votes[df_votes['is_dem'] & df_votes['is_gwannaesa']].groupby(dong_key)['득표수'].sum().reset_index(name='gwannaesa_dem')
    gn_tot = df_votes[df_votes['is_gwannaesa']].groupby(dong_key)['득표수'].sum().reset_index(name='gwannaesa_total')
    sd_dem = df_votes[df_votes['is_dem'] & ~df_votes['is_gwannaesa']].groupby(dong_key)['득표수'].sum().reset_index(name='same_day_dem')
    sd_tot = df_votes[~df_votes['is_gwannaesa']].groupby(dong_key)['득표수'].sum().reset_index(name='same_day_total')

    sum_people_dong = df_geo[~df_geo['is_gwannaesa'] & (df_geo['후보자'] == '선거인수')].groupby(dong_key)['득표수'].sum().reset_index(name='sum_people')
    sum_vote_geo = df_geo[df_geo['후보자'] == '투표수'].groupby(dong_key)['득표수'].sum().reset_index(name='sum_vote_geo')

    df_dong = gn_dem.copy()
    for frame in (gn_tot, sd_dem, sd_tot, sum_people_dong, sum_vote_geo):
        df_dong = df_dong.merge(frame, on=dong_key, how='outer')
    df_dong = df_dong.fillna(0)

    df_dong['sgg_candidates'] = df_dong['선거구명'].apply(sgg_cands_from_constituency)
    df_dong['primary_sgg']    = df_dong['sgg_candidates'].apply(lambda x: x[0] if x else "")
    df_dong['dong_norm']      = df_dong['법정읍면동명'].apply(normalize_dong_name)
    df_dong['province_tag']   = df_dong['시도명'].map(PROV_FULL_TO_SHORT).fillna(df_dong['시도명'])
    df_dong['area2_name']     = df_dong['선거구명']
    df_dong['name']           = df_dong['법정읍면동명']
    df_dong['urban_type']     = df_dong['name'].apply(get_urban_type)
    df_dong['is_military']    = df_dong.apply(check_military_zone, axis=1)

    gn_dem_c  = df_votes[df_votes['is_dem'] & df_votes['is_gwannaesa']].groupby(const_key)['득표수'].sum().reset_index(name='gwannaesa_dem')
    gn_tot_c  = df_votes[df_votes['is_gwannaesa']].groupby(const_key)['득표수'].sum().reset_index(name='gwannaesa_total')
    gn_turn_c = df_geo[df_geo['is_gwannaesa'] & (df_geo['후보자'] == '투표수')].groupby(const_key)['득표수'].sum().reset_index(name='gwannaesa_turnout')

    df_gw  = df[df['법정읍면동명'] == GWANOE_LABEL]
    df_gw_v= df_gw[~df_gw['is_meta']]
    go_dem_c  = df_gw_v[df_gw_v['is_dem']].groupby(const_key)['득표수'].sum().reset_index(name='gwanoe_dem')
    go_tot_c  = df_gw_v.groupby(const_key)['득표수'].sum().reset_index(name='gwanoe_total')
    go_turn_c = df_gw[df_gw['후보자'] == '투표수'].groupby(const_key)['득표수'].sum().reset_index(name='gwanoe_turnout')

    sd_dem_c  = df_votes[df_votes['is_dem'] & ~df_votes['is_gwannaesa']].groupby(const_key)['득표수'].sum().reset_index(name='same_day_dem')
    sd_tot_c  = df_votes[~df_votes['is_gwannaesa']].groupby(const_key)['득표수'].sum().reset_index(name='same_day_total')
    sd_turn_c = df_geo[~df_geo['is_gwannaesa'] & (df_geo['후보자'] == '투표수')].groupby(const_key)['득표수'].sum().reset_index(name='same_day_turnout')

    reg_c = df_dong.groupby(const_key)['sum_people'].sum().reset_index(name='sum_people')

    df_const = reg_c.copy()
    for frame in (gn_dem_c, gn_tot_c, gn_turn_c, go_dem_c, go_tot_c, go_turn_c, sd_dem_c, sd_tot_c, sd_turn_c):
        df_const = df_const.merge(frame, on=const_key, how='left')
    df_const = df_const.fillna(0)

    df_const['sajeong_dem']      = df_const['gwannaesa_dem']     + df_const['gwanoe_dem']
    df_const['sajeong_total']    = df_const['gwannaesa_total']   + df_const['gwanoe_total']
    df_const['sajeong_turnout']  = df_const['gwannaesa_turnout'] + df_const['gwanoe_turnout']
    df_const['total_turnout']    = df_const['sajeong_turnout']   + df_const['same_day_turnout']

    df_const['sgg_candidates'] = df_const['선거구명'].apply(sgg_cands_from_constituency)
    df_const['primary_sgg']    = df_const['sgg_candidates'].apply(lambda x: x[0] if x else "")
    df_const['province_tag']   = df_const['시도명'].map(PROV_FULL_TO_SHORT).fillna(df_const['시도명'])

    sk = ['시도명', '선거구명', '법정읍면동명', '투표구명']
    st_r = df[df['후보자'] == '선거인수'].groupby(sk)['득표수'].sum().reset_index(name='sum_people')
    st_v = df[df['후보자'] == '투표수'].groupby(sk)['득표수'].sum().reset_index(name='sum_vote')
    st_i = df[df['후보자'] == '무효 투표수'].groupby(sk)['득표수'].sum().reset_index(name='sum_invalid')
    st_d = df[df['is_dem']].groupby(sk)['득표수'].sum().reset_index(name='dem_votes')
    st_c = df[df['is_con']].groupby(sk)['득표수'].sum().reset_index(name='con_votes')

    df_station = st_r.copy()
    for frame in (st_v, st_i, st_d, st_c): df_station = df_station.merge(frame, on=sk, how='outer')
    df_station = df_station.fillna(0)

    df_station['is_early']    = df_station['투표구명'] == GWANNAESA_LABEL
    df_station['region']      = df_station['시도명'] + ' ' + df_station['선거구명']
    df_station['invalid_rate']= (df_station['sum_invalid'] / df_station['sum_vote'].replace(0, np.nan))
    
    return df_dong, df_const, df_station


# ==========================================
# 3. MULTI-PASS DONG↔CENSUS↔APT MATCHER
# ==========================================

def merge_dong_with_covariates(df_election: pd.DataFrame, df_census: pd.DataFrame, df_apt: pd.DataFrame) -> pd.DataFrame:
    if not df_census.empty:
        census_lookup = {}; census_by_sgg = {}
        for _, row in df_census.iterrows():
            dnorm = row['dong_norm']
            covs = {
                'demographic_propensity': row['demographic_propensity'],
                'pct_1824': row.get('pct_1824', np.nan),
                'pct_2529': row.get('pct_2529', np.nan),
                'pct_3034': row.get('pct_3034', np.nan),
                'pct_3539': row.get('pct_3539', np.nan),
                'pct_4044': row.get('pct_4044', np.nan),
                'pct_4549': row.get('pct_4549', np.nan),
                'pct_5054': row.get('pct_5054', np.nan),
                'pct_5559': row.get('pct_5559', np.nan),
                'pct_6064': row.get('pct_6064', np.nan),
                'pct_6569': row.get('pct_6569', np.nan),
                'pct_70plus': row.get('pct_70plus', np.nan),
                'pct_male': row.get('pct_male', np.nan),
            }
            for sgg in row['sgg_candidates']:
                census_lookup[(sgg, dnorm)] = covs
                census_by_sgg.setdefault(sgg, []).append(dnorm)

        results = []
        for _, row in df_election.iterrows():
            covs = None
            dk = row['dong_norm']
            sgc = row['sgg_candidates'] if isinstance(row['sgg_candidates'], list) else [row['primary_sgg']]

            if (row['primary_sgg'], dk) in census_lookup: covs = census_lookup[(row['primary_sgg'], dk)]
            if covs is None and '·' in dk:
                k1b = (row['primary_sgg'], dk.replace('·', ''))
                if k1b in census_lookup: covs = census_lookup[k1b]
            if covs is None:
                for sgg in sgc[1:]:
                    if (sgg, dk) in census_lookup: covs = census_lookup[(sgg, dk)]; break
            if covs is None:
                for sgg in sgc:
                    pool = census_by_sgg.get(sgg, [])
                    if pool:
                        m = get_close_matches(dk, pool, n=1, cutoff=0.82)
                        if m and (sgg, m[0]) in census_lookup: covs = census_lookup[(sgg, m[0])]; break
            
            if covs is None:
                covs = {k: np.nan for k in ['demographic_propensity', 'pct_1824', 'pct_2529', 'pct_3034', 
                                            'pct_3539', 'pct_4044', 'pct_4549', 'pct_5054', 'pct_5559', 
                                            'pct_6064', 'pct_6569', 'pct_70plus', 'pct_male']}

            rd = row.to_dict(); rd.update(covs)
            results.append(rd)
        df_out = pd.DataFrame(results)
    else:
        df_out = df_election.copy()

    # APT Imputation
    if not df_apt.empty:
        df_out = df_out.merge(df_apt, left_on=['province_tag', 'primary_sgg', 'dong_norm'], right_on=['prov', 'sgg', 'dong_norm'], how='left')
        sgg_med = df_out.groupby('primary_sgg')['median_apt_price_sqm'].transform('median')
        df_out['median_apt_price_sqm'] = df_out['median_apt_price_sqm'].fillna(sgg_med)
        prov_med = df_out.groupby('province_tag')['median_apt_price_sqm'].transform('median')
        df_out['median_apt_price_sqm'] = df_out['median_apt_price_sqm'].fillna(prov_med)
        df_out['median_apt_price_sqm'] = df_out['median_apt_price_sqm'].fillna(df_out['median_apt_price_sqm'].median())
        df_out['log_apt_price'] = np.log1p(df_out['median_apt_price_sqm'])
    else:
        df_out['log_apt_price'] = 0.0

    return df_out[df_out['pct_4044'].notna()].copy()

def merge_const_with_covariates(df_const: pd.DataFrame, df_dong_merged: pd.DataFrame) -> pd.DataFrame:
    if df_dong_merged.empty or 'pct_4044' not in df_dong_merged.columns:
        return df_const.copy()

    const_key = ['시도명', '선거구명']
    dm = df_dong_merged.dropna(subset=['pct_4044']).copy()
    
    age_cols = ['pct_1824', 'pct_2529', 'pct_3034', 'pct_3539', 'pct_4044', 'pct_4549', 
                'pct_5054', 'pct_5559', 'pct_6064', 'pct_6569', 'pct_70plus', 'pct_male']
    
    agg_funcs = {'_reg': ('sum_people', 'sum')}
    for col in age_cols:
        dm[f'_pw_{col}'] = dm[col] * dm['sum_people']
        agg_funcs[f'_pw_{col}_sum'] = (f'_pw_{col}', 'sum')

    if 'log_apt_price' in dm.columns:
        dm['_pw_log_apt_price'] = dm['log_apt_price'] * dm['sum_people']
        agg_funcs['_pw_log_apt_price_sum'] = ('_pw_log_apt_price', 'sum')
        
    agg = dm.groupby(const_key).agg(**agg_funcs).reset_index()
    
    for col in age_cols:
        agg[col] = agg[f'_pw_{col}_sum'] / agg['_reg'].replace(0, np.nan)

    if 'log_apt_price' in dm.columns:
        agg['log_apt_price'] = agg['_pw_log_apt_price_sum'] / agg['_reg'].replace(0, np.nan)
        age_cols.append('log_apt_price')

    return df_const.merge(agg[const_key + age_cols], on=const_key, how='left')


# ==========================================
# 4. FORENSICS ENGINE 
# ==========================================

def categorize_metro(region: str) -> str:
    if not isinstance(region, str): return "Other"
    if '인천' in region or '계양' in region or '연수' in region: return "Incheon"
    if any(c in region for c in ('수원', '고양', '성남', '용인')): return "Gyeonggi"
    if '서울' in region: return "Seoul"
    return "Other"

def run_forensics(df_dong_raw: pd.DataFrame, df_const_raw: pd.DataFrame, df_station: pd.DataFrame, df_census: pd.DataFrame, df_apt: pd.DataFrame) -> dict:
    logs = []
    def log(msg):
        print(msg)
        logs.append(msg)

    log(f"\n--- [4/5] Forensics Suite ---")
    MIN_VOTES = 50

    dm = merge_dong_with_covariates(df_dong_raw, df_census, df_apt)
    dm = dm[(dm['gwannaesa_total'] > MIN_VOTES) & (dm['same_day_total'] > MIN_VOTES)].copy()
    
    no_cand = (dm['gwannaesa_dem'] == 0) & (dm['same_day_dem'] == 0)
    dm = dm[~no_cand].copy()

    dm['early_pct']   = dm['gwannaesa_dem'] / dm['gwannaesa_total']
    dm['sameday_pct'] = dm['same_day_dem'] / dm['same_day_total']
    dm['gap']         = dm['early_pct'] - dm['sameday_pct']
    if 'demographic_propensity' in dm.columns:
        dm['w_gap'] = dm['gap'] / dm['demographic_propensity']
    else:
        dm['w_gap'] = dm['gap']
    dm['vote_share']  = ((dm['gwannaesa_dem'] + dm['same_day_dem']) / (dm['gwannaesa_total'] + dm['same_day_total']))
    dm['turnout']     = dm['sum_vote_geo'] / dm['sum_people'].replace(0, np.nan)

    v2bl = dm['gwannaesa_dem'].astype(int).astype(str)
    v2bl = v2bl[v2bl.str.len() >= 2]
    obs_ld  = v2bl.str[-1].astype(int).value_counts().reindex(range(10), fill_value=0).sort_index()
    exp_ld  = [len(v2bl) / 10] * 10
    variances = dm.groupby('primary_sgg')['vote_share'].std().dropna() * 100
    corr_a, _ = pearsonr(dm['sameday_pct'], dm['early_pct'])
    
    cm = merge_const_with_covariates(df_const_raw, dm)
    cm = cm[(cm['sajeong_total'] > MIN_VOTES) & (cm['same_day_total'] > MIN_VOTES)].copy()
    
    no_cand_c = (cm['sajeong_dem'] == 0) & (cm['same_day_dem'] == 0)
    cm = cm[~no_cand_c].copy()

    cm['early_pct']      = cm['sajeong_dem'] / cm['sajeong_total']
    cm['sameday_pct']    = cm['same_day_dem'] / cm['same_day_total']
    cm['gap']            = cm['early_pct'] - cm['sameday_pct']
    if 'demographic_propensity' in cm.columns:
        cm['w_gap'] = cm['gap'] / cm['demographic_propensity']
    else:
        cm['w_gap'] = cm['gap']
    cm['gwannaesa_pct']  = cm['gwannaesa_dem'] / cm['gwannaesa_total'].replace(0, np.nan)
    cm['gwanoe_pct']     = cm['gwanoe_dem'] / cm['gwanoe_total'].replace(0, np.nan)
    cm['gwannaesa_gap']  = cm['gwannaesa_pct'] - cm['sameday_pct']
    cm['gwanoe_gap']     = cm['gwanoe_pct'] - cm['sameday_pct']
    cm['gap_shift']      = cm['gap'] - cm['gwannaesa_gap']
    cm['vote_share']     = ((cm['sajeong_dem'] + cm['same_day_dem']) / (cm['sajeong_total'] + cm['same_day_total']))
    cm['turnout']        = cm['total_turnout'] / cm['sum_people'].replace(0, np.nan)

    corr_b, _ = pearsonr(cm['sameday_pct'], cm['early_pct'])

    df_station['metro_zone'] = df_station['region'].apply(categorize_metro)
    early_st  = df_station[df_station['is_early']].copy()

    return {
        'dong':  {'df': dm, 'obs_ld': obs_ld, 'exp_ld': exp_ld, 'variances': variances, 'r2': corr_a**2},
        'const': {'df': cm, 'r2': corr_b**2},
        'forensics_logs': logs
    }

# ==========================================
# 5. CAUSAL INFERENCE & REGRESSION
# ==========================================

def run_causal_analysis(dm: pd.DataFrame, cm: pd.DataFrame):
    logs = []
    def log(msg):
        print(msg)
        logs.append(msg)

    log("\n" + "="*55)
    log("  LEVEL C │ Causal Inference & Regression (5-Year Intervals)")
    log("="*55)

    age_cols = ['pct_1824', 'pct_2529', 'pct_3034', 'pct_3539', 'pct_4549', 'pct_5054', 'pct_5559', 'pct_6064', 'pct_6569', 'pct_70plus']
    
    # 1. Prepare Dong Level Model
    req_cols = ['gap', 'pct_male', 'province_tag', 'urban_type', 'is_military', 'sum_people', 'log_apt_price'] + age_cols
    df_mod = dm.dropna(subset=req_cols).copy()
    
    scaler = StandardScaler()
    cont_cols = age_cols + ['pct_male', 'log_apt_price']
    df_mod[cont_cols] = scaler.fit_transform(df_mod[cont_cols])
    
    age_formula_str = " + ".join(age_cols)
    
    formula_gap = f'gap ~ {age_formula_str} + pct_male + is_military + log_apt_price + C(province_tag) + C(urban_type)'
    model_gap = smf.wls(formula_gap, data=df_mod, weights=df_mod['sum_people']).fit(cov_type='HC3')
    log("\n[C1] Mobilization Test (WLS Gap Regression summary attached to report)")
    
    formula_ratio = f'early_pct ~ sameday_pct + {age_formula_str} + pct_male + is_military + log_apt_price + C(province_tag) + C(urban_type)'
    model_ratio = smf.wls(formula_ratio, data=df_mod, weights=df_mod['sum_people']).fit(cov_type='HC3')
    log("[C2] Algorithmic Ratio Test (WLS Ratio Regression summary attached to report)")

    df_mod['residual_gap'] = model_gap.resid
    df_mod['fitted_gap']   = model_gap.fittedvalues

    # 2. Prepare Constituency Level Model
    req_cols_c = ['gap', 'pct_male', 'province_tag', 'sum_people', 'log_apt_price'] + age_cols
    cm_mod = cm.dropna(subset=req_cols_c).copy()
    if not cm_mod.empty:
        cm_mod[cont_cols] = scaler.fit_transform(cm_mod[cont_cols])
        formula_gap_c = f'gap ~ {age_formula_str} + pct_male + log_apt_price + C(province_tag)'
        model_gap_c = smf.wls(formula_gap_c, data=cm_mod, weights=cm_mod['sum_people']).fit(cov_type='HC3')
        cm_mod['residual_gap'] = model_gap_c.resid

    # 3. Propensity Score Calculation
    median_turnout = df_mod['gwannaesa_total'].median()
    df_mod['D_high_early'] = (df_mod['gwannaesa_total'] > median_turnout).astype(int)

    urban_dummies = pd.get_dummies(df_mod['urban_type'], drop_first=True)
    prov_dummies = pd.get_dummies(df_mod['province_tag'], drop_first=True)
    
    X = pd.concat([df_mod[age_cols + ['pct_male', 'is_military', 'log_apt_price']], urban_dummies, prov_dummies], axis=1)
    y = df_mod['D_high_early']
    
    lr = LogisticRegression(solver='liblinear', max_iter=500)
    lr.fit(X, y)
    df_mod['propensity_score'] = lr.predict_proba(X)[:, 1]

    treated = df_mod[df_mod['D_high_early'] == 1]
    control = df_mod[df_mod['D_high_early'] == 0]
    
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(control[['propensity_score']])
    distances, indices = nn.kneighbors(treated[['propensity_score']])
    
    matched_control = control.iloc[indices.flatten()]
    
    treated_mean = treated['vote_share'].mean()
    matched_control_mean = matched_control['vote_share'].mean()
    att = treated_mean - matched_control_mean
    
    log("\n[C3] Propensity Score Matching (ATT)")
    log(f"Treatment: High Early Turnout Volume (> {median_turnout:,.0f} votes)")
    log(f"Matched on: 5-Year Age Bins, Gender, Urbanization, Wealth, Region, Military")
    log(f"  Treated Dem Vote Share Mean: {treated_mean*100:.2f}%")
    log(f"  Matched Control Share Mean : {matched_control_mean*100:.2f}%")
    log(f"  Estimated ATT              : {att*100:+.2f}%")

    raw_mean = df_mod['gap'].mean()
    r_squared = model_gap.rsquared
    
    log("\n[C4] Fraud vs. Mobilization Diagnostic (WLS Variance Analysis)")
    log(f"  Raw Mean Gap                     : {raw_mean*100:+.2f}%")
    log(f"  Demographic Explained Variance R²: {r_squared*100:.2f}%")
    log(f"  Unexplained Variance             : {(1 - r_squared)*100:.2f}%")

    if r_squared < 0.10:
        log("  Conclusion: High-resolution demographics explain very little (<10%) of the early voting gap.")
        log("              The gap operates largely independently of these known covariates.")
        log("              This leaves room for unobserved covariates or aligns with artificial uniform shifts.")
    elif r_squared >= 0.30:
        log("  Conclusion: High-resolution demographics explain a massive portion (>30%) of the early voting gap variance.")
        log("              Strong evidence FOR organic demographic sorting (Mobilization).")
        log("              Weakens theories of a uniform, demographic-blind algorithmic fraud.")

    res_data = df_mod['residual_gap'].replace([np.inf, -np.inf], np.nan).dropna()
    stat, p_norm = normaltest(res_data)
    log("\n[C5] Residual Normality Diagnostic (D'Agostino's K-squared test)")
    log(f"  Test Statistic : {stat:.4f}")
    log(f"  p-value        : {p_norm:.4e}")

    log("\n[C6] Anomaly Profiling: Top 8 Dongs by Absolute Unexplained Residual (Pop > 2,000)")
    log(f"{'Region':<16} {'Dong':<14} {'Raw Gap':>9} {'Fitted':>9} {'Residual':>9}")
    
    valid_dongs = df_mod[df_mod['sum_people'] >= 2000]
    top_outliers = valid_dongs.reindex(valid_dongs['residual_gap'].abs().sort_values(ascending=False).index).head(8)
    for _, row in top_outliers.iterrows():
        region = f"{row['province_tag']} {row['area2_name']}"[:15]
        dong = str(row['name'])[:13]
        r_gap = f"{row['gap']*100:+.2f}%"
        f_gap = f"{row['fitted_gap']*100:+.2f}%"
        resid = f"{row['residual_gap']*100:+.2f}%"
        log(f"  {region:<16} {dong:<14} {r_gap:>9} {f_gap:>9} {resid:>9}")

    return df_mod, cm_mod if not cm_mod.empty else cm, model_gap, model_ratio, logs

# ==========================================
# 6. DASHBOARDS & REPORTS
# ==========================================
def plot_dashboard(results: dict, out_path: str, title: str):
    print(f"\nGenerating visual dashboard → {out_path}")
    dm       = results['dong']['df']
    cm       = results['const']['df']
    obs_ld   = results['dong']['obs_ld']
    exp_ld   = results['dong']['exp_ld']
    variances= results['dong']['variances']
    r2_dong  = results['dong']['r2']
    r2_const = results['const']['r2']

    fig, axes = plt.subplots(4, 4, figsize=(22, 20))
    fig.suptitle(f"Election Forensics & Causal Dashboard  –  {title}", fontsize=15, fontweight='bold', y=0.995)

    LEVEL_A = '관내사전 vs 본투표 (dong)'
    LEVEL_B = '사전 전체 vs 본투표 (constituency)'

    def hist_gap(ax, data, label, color):
        data = data.replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) == 0: return
        ax.hist(data * 100, bins=40, color=color, alpha=0.65, edgecolor='none')
        ax.axvline(data.mean() * 100, color='black', lw=1.8, ls='--',
                   label=f'μ = {data.mean()*100:+.2f}%\nσ = {data.std()*100:.2f}%')
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Gap (%)', fontsize=8)
        ax.legend(fontsize=7, handlelength=1)

    def scatter_ratio(ax, x, y, r2, label):
        if len(x) == 0 or len(y) == 0: return
        mn = min(x.min(), y.min()) * 100 - 1
        mx = max(x.max(), y.max()) * 100 + 1
        ax.scatter(x * 100, y * 100, alpha=0.25, s=12, color='royalblue', edgecolors='none')
        ax.plot([mn, mx], [mn, mx], 'r--', lw=1.2, label='1:1')
        ax.set_title(f'{label}  R²={r2:.4f}', fontsize=9)
        ax.set_xlabel('본투표 dem share (%)', fontsize=8)
        ax.set_ylabel('사전 dem share (%)', fontsize=8)
        ax.legend(fontsize=7)

    def plot_residual_hist(ax, data, label, color):
        data = data.replace([np.inf, -np.inf], np.nan).dropna() * 100
        if len(data) == 0: 
            ax.axis('off')
            return
        count, bins, ignored = ax.hist(data, bins=40, density=True, color=color, alpha=0.5)
        mu, std = data.mean(), data.std()
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        ax.plot(x, p, 'k', linewidth=2, label=f'Fit $\\mu$={mu:.2f}, $\\sigma$={std:.2f}')
        ax.axvline(0, color='red', lw=1, ls='--', alpha=0.5)
        ax.set_title(f'WLS Residuals: {label}', fontsize=9)
        ax.legend(fontsize=7)

    def plot_resid_vs_fitted(ax, df_mod, label):
        if 'fitted_gap' not in df_mod.columns or 'residual_gap' not in df_mod.columns:
            ax.axis('off')
            return
        x = df_mod['fitted_gap'] * 100
        y = df_mod['residual_gap'] * 100
        if len(x) == 0: return
        ax.scatter(x, y, alpha=0.25, s=12, color='darkmagenta', edgecolors='none')
        ax.axhline(0, color='red', lw=1.2, ls='--')
        ax.set_title(f'Residuals vs Fitted\n{label}', fontsize=9)
        ax.set_xlabel('Fitted Gap (%)', fontsize=8)
        ax.set_ylabel('Residual Gap (%)', fontsize=8)

    def plot_qq(ax, data, title_str):
        data = data.replace([np.inf, -np.inf], np.nan).dropna() * 100
        if len(data) == 0:
            ax.axis('off')
            return
        probplot(data, dist="norm", plot=ax)
        ax.set_title(f'Q-Q Plot (WLS Residuals): {title_str}', fontsize=9)
        ax.get_lines()[0].set_markerfacecolor('steelblue')
        ax.get_lines()[0].set_markeredgecolor('none')
        ax.get_lines()[0].set_markersize(4)
        ax.get_lines()[1].set_color('red')
        ax.set_xlabel('Theoretical Quantiles (Normal)', fontsize=8)
        ax.set_ylabel('Ordered Residuals', fontsize=8)

    hist_gap(axes[0, 0], dm['gap'], f'Raw gap\n{LEVEL_A}', 'tomato')
    hist_gap(axes[0, 1], cm['gap'], f'Raw gap\n{LEVEL_B}', 'darkorange')
    hist_gap(axes[0, 2], dm.get('w_gap', dm['gap']), f'Orig Weighted gap\n{LEVEL_A}', 'seagreen')
    hist_gap(axes[0, 3], cm.get('w_gap', cm['gap']), f'Orig Weighted gap\n{LEVEL_B}', 'teal')

    scatter_ratio(axes[1, 0], dm['sameday_pct'], dm['early_pct'], r2_dong, f'Algorithmic ratio\n{LEVEL_A}')
    scatter_ratio(axes[1, 1], cm['sameday_pct'], cm['early_pct'], r2_const, f'Algorithmic ratio\n{LEVEL_B}')

    fp = dm[(dm['turnout'] > 0) & (dm['turnout'] <= 1.0)]
    hb = axes[1, 2].hexbin(fp['turnout']*100, fp['vote_share']*100, gridsize=28, cmap='inferno', mincnt=1)
    fig.colorbar(hb, ax=axes[1, 2], label='dongs')
    axes[1, 2].set_title(f'Election fingerprint\n{LEVEL_A}', fontsize=9)

    fpc = cm[(cm['turnout'] > 0) & (cm['turnout'] <= 1.0)]
    hb2 = axes[1, 3].hexbin(fpc['turnout']*100, fpc['vote_share']*100, gridsize=18, cmap='inferno', mincnt=1)
    fig.colorbar(hb2, ax=axes[1, 3], label='const.')
    axes[1, 3].set_title(f'Election fingerprint\n{LEVEL_B}', fontsize=9)

    axes[2, 0].bar(range(10), obs_ld, color='salmon', alpha=0.8, edgecolor='white', label='Observed')
    axes[2, 0].plot(range(10), exp_ld, 'k--', lw=1.5, label='Expected uniform')
    axes[2, 0].set_title(f'Last-digit test\n관내사전 dem votes (dong)', fontsize=9)
    axes[2, 0].set_xticks(range(10)); axes[2, 0].legend(fontsize=7)

    axes[2, 1].hist(variances, bins=25, color='mediumpurple', alpha=0.75, edgecolor='none')
    axes[2, 1].set_title(f'Vote-share variance by city\n{LEVEL_A}', fontsize=9)

    vgo = cm.dropna(subset=['gwanoe_pct'])
    axes[2, 2].scatter(vgo['gwannaesa_gap']*100, vgo['gwanoe_gap']*100, alpha=0.35, s=14, color='steelblue', edgecolors='none')
    lims = [min(vgo['gwannaesa_gap'].min(), vgo['gwanoe_gap'].min())*100 - 1,
            max(vgo['gwannaesa_gap'].max(), vgo['gwanoe_gap'].max())*100 + 1]
    axes[2, 2].plot(lims, lims, 'r--', lw=1.2, label='1:1')
    axes[2, 2].set_title('관내사전 gap vs 관외사전 gap\n(both vs 본투표, constituency)', fontsize=9)
    axes[2, 2].legend(fontsize=7)

    hist_gap(axes[2, 3], cm['gap_shift'], 'Gap shift from adding 관외사전\n(constituency)', 'goldenrod')

    plot_residual_hist(axes[3, 0], dm.get('residual_gap', pd.Series(dtype=float)), f'{LEVEL_A}', 'mediumaquamarine')
    
    if 'residual_gap' in cm.columns:
        plot_residual_hist(axes[3, 1], cm.get('residual_gap', pd.Series(dtype=float)), f'{LEVEL_B}', 'cadetblue')
    else:
        axes[3, 1].axis('off')
        
    plot_resid_vs_fitted(axes[3, 2], dm, f'{LEVEL_A}')
    plot_qq(axes[3, 3], dm.get('residual_gap', pd.Series(dtype=float)), f'{LEVEL_A}')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=200)
    print(f"Saved visual dashboard → '{out_path}'")


def plot_statistical_report(results: dict, out_path: str, title: str):
    print(f"Generating statistical report image → {out_path}")
    # Increased figsize significantly to accommodate the massive 5-year interval regression table
    fig, ax = plt.subplots(figsize=(14, 24))
    ax.axis('off')
    
    text = f"=== STATISTICAL DIAGNOSTIC REPORT: {title} ===\n\n"
    text += "\n".join(results['forensics_logs']) + "\n\n"
    text += "="*60 + "\n"
    text += "=== CAUSAL INFERENCE & WLS REGRESSION ===\n"
    text += "="*60 + "\n\n"
    text += "\n".join(results['causal_logs']) + "\n\n"
    text += "[C1] Mobilization Test (WLS Gap Regression)\n"
    text += results['mod_gap'].summary().as_text() + "\n\n"
    text += "[C2] Algorithmic Ratio Test (WLS Ratio Regression)\n"
    text += results['mod_ratio'].summary().as_text() + "\n"
    
    # Increased font size to 8.5 for better readability in the final image
    ax.text(0.01, 0.99, text, fontsize=8.5, va='top', ha='left')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved statistical report → '{out_path}'")


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Korean Election Forensics — {CFG['label']}")
    print(f"{'='*60}")

    setup_korean_font()

    df_census = load_census_csv(CFG['census_csv'])
    df_apt = load_apt_csv(CFG['apt_csv_glob'])
    df_dong, df_const, df_station = load_election_csv(
        CFG['result_csv'], dem_pattern=CFG['dem_pattern'], con_pattern=CFG['con_pattern']
    )

    if not df_dong.empty and not df_const.empty:
        results = run_forensics(df_dong, df_const, df_station, df_census, df_apt)
        
        dm_raw = results['dong']['df']
        cm_raw = results['const']['df']
        dm_causal, cm_causal, mod_gap, mod_ratio, causal_logs = run_causal_analysis(dm_raw, cm_raw)
        
        results['dong']['df'] = dm_causal
        results['const']['df'] = cm_causal
        results['mod_gap'] = mod_gap
        results['mod_ratio'] = mod_ratio
        results['causal_logs'] = causal_logs

        plot_dashboard(results, out_path=CFG['dashboard_out'], title=CFG['label'])
        report_out = CFG['dashboard_out'].replace('.png', '_report.png')
        plot_statistical_report(results, out_path=report_out, title=CFG['label'])
        
    else:
        print("[!] Election data could not be loaded. Aborting.")

import pandas as pd
import numpy as np
import math
import os
import re
import glob
from difflib import get_close_matches
from scipy.stats import norm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import koreanize_matplotlib
except ImportError:
    print("[!] koreanize_matplotlib not found. Run: pip install koreanize-matplotlib")

# ==========================================
# CONFIGURATION
# ==========================================

# Election configurations — keyed by short ID
# census_key: which election's census/apt data to borrow when no dedicated file exists
ELECTION_CONFIGS = {
    'gen21': {
        'type':         'general',
        'result_csv':   '21st_election_result.csv',
        'census_csv':   '21st_election_census.csv',
        'census_key':   'gen21',           # own data
        'apt_csv_glob': '*21st_election_*_apt_price.csv',
        'dem_pattern':  r'더불어민주당',
        'label':        '21st General (2020)',
        'year':         2020,
    },
    'gen22': {
        'type':         'general',
        'result_csv':   '22nd_election_result.csv',
        'census_csv':   '22nd_election_census.csv',
        'census_key':   'gen22',           # own data
        'apt_csv_glob': '*22nd_election_*_apt_price.csv',
        'dem_pattern':  r'더불어민주당',
        'label':        '22nd General (2024)',
        'year':         2024,
    },
    'pres20': {
        'type':         'presidential',
        'result_csv':   '20th_presidential_election_result.csv',
        'census_key':   'gen21',           # borrow from nearest general (2020)
        'apt_csv_glob': '*pres20*_apt_price.csv',
        'dem_pattern':  r'더불어민주당',
        'label':        '20th Presidential (2022)',
        'year':         2022,
    },
    'pres21': {
        'type':         'presidential',
        'result_csv':   '21st_presidential_election_result.csv',
        'census_key':   'gen22',           # borrow from nearest general (2024)
        'apt_csv_glob': '*pres21*_apt_price.csv',
        'dem_pattern':  r'더불어민주당',
        'label':        '21st Presidential (2025)',
        'year':         2025,
    },
}

# Five comparison pairs:
#   (baseline_key, comparison_key, human_label, series_tag)
# series_tag groups pairs for multi-panel layout
COMPARISON_PAIRS = [
    ('gen21',  'gen22',  'General → General: 21st (2020) → 22nd (2024)',           'General-to-General'),
    ('pres20', 'pres21', 'Presidential → Presidential: 20th (2022) → 21st (2025)', 'Presidential-to-Presidential'),
    ('gen21',  'pres20', 'Nearest: 21st General (2020) → 20th Presidential (2022)','Nearest-Adjacent'),
    ('pres20', 'gen22',  'Nearest: 20th Presidential (2022) → 22nd General (2024)','Nearest-Adjacent'),
    ('gen22',  'pres21', 'Nearest: 22nd General (2024) → 21st Presidential (2025)','Nearest-Adjacent'),
]

# ==========================================
# POOL CONFIGURATIONS  (N-election pools)
# ==========================================
# Each pool specifies:
#   keys      : ordered list of election keys to pool (first = reference)
#   label     : human title for the analysis
#   out_prefix: prefix for output filenames
POOL_CONFIGS = [
    {
        'keys':       ['pres20', 'pres21'],
        'label':      'Pure Presidential Pool: 20th (2022) vs 21st (2025)',
        'out_prefix': 'pool_presidential',
    },
    {
        'keys':       ['gen21', 'gen22'],
        'label':      'Pure General Election Pool: 21st (2020) vs 22nd (2024)',
        'out_prefix': 'pool_general',
    },
    {
        'keys':       ['gen21', 'gen22', 'pres20'],
        'label':      'General + 20th Presidential Pool: 21st (2020) / 22nd (2024) / 20th Pres (2022)',
        'out_prefix': 'pool_gen_pres20',
    },
    {
        'keys':       ['gen21', 'gen22', 'pres20', 'pres21'],
        'label':      'All Elections Pool: 21st Gen / 22nd Gen / 20th Pres / 21st Pres',
        'out_prefix': 'pool_all',
    },
]

# ==========================================
# SINGLE-ELECTION CONFIGURATIONS
# ==========================================
# Run a standalone cross-sectional WLS for each election individually.
# Produces absolute coefficient estimates (no differencing).
SINGLE_CONFIGS = [
    {'key': 'gen21',  'out_prefix': 'single_gen21'},
    {'key': 'gen22',  'out_prefix': 'single_gen22'},
    {'key': 'pres20', 'out_prefix': 'single_pres20'},
    {'key': 'pres21', 'out_prefix': 'single_pres21'},
]



# ==========================================
# CONSTANTS
# ==========================================
SPECIAL_DONG_NAMES_GENERAL = {
    '거소·선상투표', '관외사전투표', '국외부재자투표',
    '국외부재자투표(공관)', '잘못 투입·구분된 투표지',
}
SPECIAL_DONG_NAMES_PRES = {
    '거소·선상투표', '관외사전투표', '재외투표',
    '잘못 투입·구분된 투표지',
}

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

AGE_GENDER_COLS = [
    'pct_m_1824', 'pct_m_2529', 'pct_m_3034', 'pct_m_3539', 'pct_m_4044', 'pct_m_4549',
    'pct_m_5054', 'pct_m_5559', 'pct_m_6064', 'pct_m_6569', 'pct_m_70plus',
    'pct_f_1824', 'pct_f_2529', 'pct_f_3034', 'pct_f_3539', 'pct_f_4044', 'pct_f_4549',
    'pct_f_5054', 'pct_f_5559', 'pct_f_6064', 'pct_f_6569', 'pct_f_70plus'
]

# ==========================================
# SHARED UTILITIES
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
    gu_count     = sum(1 for t in tokens if t[-1] == '구'  and len(t) >= 2)
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
    sgg  = str(row.get('area2_name', ''))
    dong = str(row.get('name', ''))
    military_dongs = [
        '진동면','군내면','장단면','파평면','중면','장남면','백학면','왕징면',
        '근북면','근동면','원동면','원남면','임남면','동송읍','철원읍',
        '상서면','서화면','방산면','해안면','현내면',
        '백령면','대청면','연평면','신도안면','오천읍','고경면',
    ]
    if any(m in dong for m in military_dongs): return 1
    if '진해구' in sgg: return 1
    return 0

# ==========================================
# DATA LOADERS
# ==========================================
def _detect_year_prefix(df: pd.DataFrame) -> str:
    for col in df.columns:
        m = re.match(r'(\d{4}년\d{2}월)_계_총인구수', col)
        if m: return m.group(1)
    # Help the user diagnose column naming issues
    sample = [c for c in df.columns[:20]]
    raise ValueError(
        f"Cannot detect census year prefix (expected 'YYYY년MM월_계_총인구수').\n"
        f"First 20 columns seen: {sample}"
    )

def load_census_csv(csv_path: str) -> pd.DataFrame:
    resolved = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        print(f"  [!] Census file not found: {resolved}")
        return pd.DataFrame()
    print(f"  [Census] Reading: {resolved}")
    try:
        try:   df = pd.read_csv(csv_path, encoding='utf-8',  low_memory=False)
        except UnicodeDecodeError:
               df = pd.read_csv(csv_path, encoding='cp949', low_memory=False)

        prefix = _detect_year_prefix(df)
        voting_age_cols = (
            [f"{prefix}_계_{a}세" for a in range(18, 100)] +
            [f"{prefix}_계_100세 이상"]
        )
        all_target_cols = list(voting_age_cols)
        for g in ['남', '여']:
            for a in range(18, 100):
                all_target_cols.append(f"{prefix}_{g}_{a}세")
            all_target_cols.append(f"{prefix}_{g}_100세 이상")

        for col in set(all_target_cols):
            if col in df.columns:
                df[col] = (df[col].astype(str)
                           .str.replace(',', '', regex=False)
                           .pipe(pd.to_numeric, errors='coerce')
                           .fillna(0))

        df['total_voting_pop'] = df[[c for c in voting_age_cols if c in df.columns]].sum(axis=1)
        df = df[df['total_voting_pop'] > 0].copy()

        ranges = [
            (18,25,'1824'),(25,30,'2529'),(30,35,'3034'),(35,40,'3539'),
            (40,45,'4044'),(45,50,'4549'),(50,55,'5054'),(55,60,'5559'),
            (60,65,'6064'),(65,70,'6569'),
        ]
        for g, g_str in [('남','m'), ('여','f')]:
            for r_s, r_e, r_str in ranges:
                cols = [f"{prefix}_{g}_{a}세" for a in range(r_s, r_e)]
                df[f'pct_{g_str}_{r_str}'] = (
                    df[[c for c in cols if c in df.columns]].sum(axis=1)
                    / df['total_voting_pop']
                )
            cols_70 = [f"{prefix}_{g}_{a}세" for a in range(70,100)] + [f"{prefix}_{g}_100세 이상"]
            df[f'pct_{g_str}_70plus'] = (
                df[[c for c in cols_70 if c in df.columns]].sum(axis=1)
                / df['total_voting_pop']
            )

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
            row_dict = {
                'sgg_candidates': sgg_cands,
                'primary_sgg':    sgg_cands[0] if sgg_cands else "",
                'dong_norm':      dong_norm,
                'dong_raw':       row['행정구역'],
            }
            for col in AGE_GENDER_COLS:
                row_dict[col] = row.get(col, np.nan)
            rows.append(row_dict)
        return pd.DataFrame(rows)
    except Exception as e:
        import traceback
        print(f"[!] Error processing census CSV {csv_path}: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def load_apt_csv(glob_pattern: str) -> pd.DataFrame:
    file_list = glob.glob(glob_pattern)
    if not file_list: return pd.DataFrame()
    df_list = []
    for file in file_list:
        try:
            try:   df_temp = pd.read_csv(file, encoding='utf-8',  skiprows=15)
            except UnicodeDecodeError:
                   df_temp = pd.read_csv(file, encoding='cp949', skiprows=15)
            df_list.append(df_temp)
        except Exception:
            pass
    if not df_list: return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True)
    try:
        df['거래금액(만원)']  = pd.to_numeric(
            df['거래금액(만원)'].astype(str).str.replace(',','').str.strip(), errors='coerce')
        df['전용면적(㎡)']    = pd.to_numeric(df['전용면적(㎡)'], errors='coerce')
        df['price_per_sqm'] = df['거래금액(만원)'] / df['전용면적(㎡)']

        def parse_loc(x):
            parts = str(x).split()
            prov = PROV_FULL_TO_SHORT.get(parts[0], parts[0]) if parts else ""
            sgg  = (normalize_sigungu(parts[1])[0]
                    if len(parts) > 2 and normalize_sigungu(parts[1]) else "")
            dong = normalize_dong_name(parts[-1]) if parts else ""
            return pd.Series([prov, sgg, dong])

        df[['prov','sgg','dong_norm']] = df['시군구'].apply(parse_loc)
        apt_agg = (df.groupby(['prov','sgg','dong_norm'])['price_per_sqm']
                     .median().reset_index()
                     .rename(columns={'price_per_sqm': 'median_apt_price_sqm'}))
        return apt_agg
    except Exception:
        return pd.DataFrame()

# ---- General election loader (unchanged logic, slightly refactored) ----
def load_general_election_csv(csv_path: str, dem_pattern: str) -> pd.DataFrame:
    try:
        try:   df = pd.read_csv(csv_path, encoding='utf-8',  low_memory=False)
        except UnicodeDecodeError:
               df = pd.read_csv(csv_path, encoding='cp949', low_memory=False)
    except Exception as e:
        print(f"[!] Cannot read {csv_path}: {e}")
        return pd.DataFrame()

    df['득표수']  = pd.to_numeric(df['득표수'], errors='coerce').fillna(0).astype(int)
    df['is_dem']  = df['후보자'].str.contains(dem_pattern, case=False, na=False)
    df['is_meta'] = df['후보자'].isin(META_CANDIDATES)

    dong_key = ['시도명', '선거구명', '법정읍면동명']
    df_geo   = df[~df['법정읍면동명'].isin(SPECIAL_DONG_NAMES_GENERAL)].copy()
    df_votes = df_geo[~df_geo['is_meta']].copy()

    total_dem   = df_votes[df_votes['is_dem']].groupby(dong_key)['득표수'].sum().reset_index(name='total_dem')
    total_votes = df_votes.groupby(dong_key)['득표수'].sum().reset_index(name='total_votes')
    sum_people  = df_geo[df_geo['후보자'] == '선거인수'].groupby(dong_key)['득표수'].sum().reset_index(name='sum_people')

    df_dong = total_dem.copy()
    for frame in (total_votes, sum_people):
        df_dong = df_dong.merge(frame, on=dong_key, how='outer')
    df_dong = df_dong.fillna(0)

    def sgg_from_constituency(name):
        if not isinstance(name, str): return []
        if '_' in name: return normalize_sigungu(name.split('_', 1)[1])
        return normalize_sigungu(re.sub(r'[갑을병정무]$', '', name).strip())

    df_dong['sgg_candidates'] = df_dong['선거구명'].apply(sgg_from_constituency)
    df_dong['primary_sgg']    = df_dong['sgg_candidates'].apply(lambda x: x[0] if x else "")
    df_dong['dong_norm']      = df_dong['법정읍면동명'].apply(normalize_dong_name)
    df_dong['province_tag']   = df_dong['시도명'].map(PROV_FULL_TO_SHORT).fillna(df_dong['시도명'])
    df_dong['area2_name']     = df_dong['선거구명']
    df_dong['name']           = df_dong['법정읍면동명']
    df_dong['urban_type']     = df_dong['name'].apply(get_urban_type)
    df_dong['is_military']    = df_dong.apply(check_military_zone, axis=1)
    return df_dong

# ---- Presidential election loader (new) ----
def load_presidential_election_csv(csv_path: str, dem_pattern: str) -> pd.DataFrame:
    """
    Handles NEC presidential election CSV format:
      시도명 | 구시군명 | 읍면동명 | 투표구명 | 후보자 | 득표수
    """
    try:
        try:   df = pd.read_csv(csv_path, encoding='utf-8-sig', low_memory=False)
        except UnicodeDecodeError:
               df = pd.read_csv(csv_path, encoding='cp949',     low_memory=False)
    except Exception as e:
        print(f"[!] Cannot read {csv_path}: {e}")
        return pd.DataFrame()

    df['득표수']  = pd.to_numeric(df['득표수'], errors='coerce').fillna(0).astype(int)
    df['is_dem']  = df['후보자'].str.contains(dem_pattern, case=False, na=False)
    df['is_meta'] = df['후보자'].isin(META_CANDIDATES)

    dong_key = ['시도명', '구시군명', '읍면동명']
    df_geo   = df[~df['읍면동명'].isin(SPECIAL_DONG_NAMES_PRES)].copy()
    # Also drop rows where 읍면동명 contains '재외' (overseas voting booths)
    df_geo   = df_geo[~df_geo['읍면동명'].str.contains('재외', na=False)].copy()
    df_votes = df_geo[~df_geo['is_meta']].copy()

    total_dem   = df_votes[df_votes['is_dem']].groupby(dong_key)['득표수'].sum().reset_index(name='total_dem')
    total_votes = df_votes.groupby(dong_key)['득표수'].sum().reset_index(name='total_votes')
    sum_people  = df_geo[df_geo['후보자'] == '선거인수'].groupby(dong_key)['득표수'].sum().reset_index(name='sum_people')

    df_dong = total_dem.copy()
    for frame in (total_votes, sum_people):
        df_dong = df_dong.merge(frame, on=dong_key, how='outer')
    df_dong = df_dong.fillna(0)

    # For presidential: 구시군명 IS the SGU directly, no constituency encoding needed
    def sgg_from_sigungu(name):
        return normalize_sigungu(str(name)) if isinstance(name, str) else []

    df_dong['sgg_candidates'] = df_dong['구시군명'].apply(sgg_from_sigungu)
    df_dong['primary_sgg']    = df_dong['sgg_candidates'].apply(lambda x: x[0] if x else "")
    df_dong['dong_norm']      = df_dong['읍면동명'].apply(normalize_dong_name)
    df_dong['province_tag']   = df_dong['시도명'].map(PROV_FULL_TO_SHORT).fillna(df_dong['시도명'])
    df_dong['area2_name']     = df_dong['구시군명']
    df_dong['name']           = df_dong['읍면동명']
    df_dong['urban_type']     = df_dong['name'].apply(get_urban_type)
    df_dong['is_military']    = df_dong.apply(check_military_zone, axis=1)
    return df_dong

def load_election_csv(cfg: dict) -> pd.DataFrame:
    """Dispatcher: routes to general or presidential loader based on config."""
    if cfg['type'] == 'general':
        return load_general_election_csv(cfg['result_csv'], cfg['dem_pattern'])
    else:
        return load_presidential_election_csv(cfg['result_csv'], cfg['dem_pattern'])

# ==========================================
# COVARIATE MERGE
# ==========================================
def merge_dong_with_covariates(
    df_election: pd.DataFrame,
    df_census:   pd.DataFrame,
    df_apt:      pd.DataFrame,
) -> pd.DataFrame:

    census_available = not df_census.empty

    if census_available:
        census_lookup  = {}
        census_by_sgg  = {}
        for _, row in df_census.iterrows():
            dnorm = row['dong_norm']
            covs  = {c: row.get(c, np.nan) for c in AGE_GENDER_COLS}
            for sgg in row['sgg_candidates']:
                census_lookup[(sgg, dnorm)] = covs
                census_by_sgg.setdefault(sgg, []).append(dnorm)

        results = []
        for _, row in df_election.iterrows():
            covs  = None
            dk    = row['dong_norm']
            sgc   = (row['sgg_candidates']
                     if isinstance(row['sgg_candidates'], list)
                     else [row['primary_sgg']])

            if (row['primary_sgg'], dk) in census_lookup:
                covs = census_lookup[(row['primary_sgg'], dk)]
            if covs is None and '·' in dk:
                k1b = (row['primary_sgg'], dk.replace('·', ''))
                if k1b in census_lookup: covs = census_lookup[k1b]
            if covs is None:
                for sgg in sgc[1:]:
                    if (sgg, dk) in census_lookup:
                        covs = census_lookup[(sgg, dk)]; break
            if covs is None:
                for sgg in sgc:
                    pool = census_by_sgg.get(sgg, [])
                    if pool:
                        m = get_close_matches(dk, pool, n=1, cutoff=0.82)
                        if m and (sgg, m[0]) in census_lookup:
                            covs = census_lookup[(sgg, m[0])]; break

            if covs is None:
                covs = {k: np.nan for k in AGE_GENDER_COLS}
            rd = row.to_dict(); rd.update(covs)
            results.append(rd)
        df_out = pd.DataFrame(results)
    else:
        # No census: keep all rows; age/gender columns will be absent / all-NaN
        df_out = df_election.copy()
        for col in AGE_GENDER_COLS:
            df_out[col] = np.nan

    if not df_apt.empty:
        df_out = df_out.merge(
            df_apt,
            left_on=['province_tag', 'primary_sgg', 'dong_norm'],
            right_on=['prov', 'sgg', 'dong_norm'],
            how='left',
        )
        sgg_med  = df_out.groupby('primary_sgg')['median_apt_price_sqm'].transform('median')
        df_out['median_apt_price_sqm'] = df_out['median_apt_price_sqm'].fillna(sgg_med)
        prov_med = df_out.groupby('province_tag')['median_apt_price_sqm'].transform('median')
        df_out['median_apt_price_sqm'] = df_out['median_apt_price_sqm'].fillna(prov_med)
        df_out['median_apt_price_sqm'] = df_out['median_apt_price_sqm'].fillna(
            df_out['median_apt_price_sqm'].median())
        df_out['log_apt_price'] = np.log1p(df_out['median_apt_price_sqm'])
    else:
        df_out['log_apt_price'] = 0.0

    # Only discard rows with missing census data if census was actually provided.
    # When census is unavailable, keep all rows and let the regression adapt.
    if census_available:
        df_out = df_out[df_out['pct_f_4044'].notna()].copy()

    df_out['_has_census'] = census_available
    return df_out.copy()

# ==========================================
# REGRESSION ENGINE — 2-election pair
# ==========================================
def run_pooled_shift_regression(
    dm_base:    pd.DataFrame,
    dm_comp:    pd.DataFrame,
    label_base: str,
    label_comp: str,
) -> dict:
    """Binary DiD for a single election pair (is_comp = 0 / 1)."""
    MIN_VOTES = 50
    dm_b = dm_base[(dm_base['total_votes'] > MIN_VOTES) & (dm_base['total_dem'] > 0)].copy()
    dm_b['is_comp'] = 0
    dm_c = dm_comp[(dm_comp['total_votes'] > MIN_VOTES) & (dm_comp['total_dem'] > 0)].copy()
    dm_c['is_comp'] = 1

    df_pooled = pd.concat([dm_b, dm_c], ignore_index=True)
    df_pooled['vote_share'] = df_pooled['total_dem'] / df_pooled['total_votes']

    potential_age_cols = [c for c in AGE_GENDER_COLS if c != 'pct_f_4044']
    coverage = {c: df_pooled[c].notna().mean()
                for c in potential_age_cols if c in df_pooled.columns}
    model_age_gender_cols = [c for c, cov in coverage.items() if cov >= 0.50]

    if model_age_gender_cols:
        print(f"  Census covariates: {len(model_age_gender_cols)} age/gender columns.")
    else:
        print("  [!] No census data — reduced model.")

    base_req = ['vote_share', 'is_comp', 'province_tag', 'urban_type',
                'is_military', 'sum_people', 'log_apt_price']
    df_mod = df_pooled.dropna(subset=base_req).copy()
    if model_age_gender_cols:
        df_mod = df_mod.dropna(subset=model_age_gender_cols).copy()

    cont_cols = model_age_gender_cols + ['log_apt_price']
    scaler = StandardScaler()
    if cont_cols:
        df_mod[cont_cols] = scaler.fit_transform(df_mod[cont_cols])

    age_part        = (' + '.join(model_age_gender_cols) + ' + ') if model_age_gender_cols else ''
    base_covariates = (f"{age_part}is_military + log_apt_price "
                       f"+ C(province_tag) + C(urban_type)")
    if model_age_gender_cols:
        cohort_shift = ' + '.join([f"is_comp:{c}" for c in model_age_gender_cols])
        formula = (f"vote_share ~ {base_covariates} + is_comp "
                   f"+ is_comp:C(province_tag) + {cohort_shift}")
    else:
        formula = f"vote_share ~ {base_covariates} + is_comp + is_comp:C(province_tag)"

    print(f"  Fitting: {label_base} (0) vs {label_comp} (1) ...")
    model = smf.wls(formula, data=df_mod, weights=df_mod['sum_people']).fit(cov_type='HC3')
    print(f"  R²: {model.rsquared*100:.2f}%  |  N: {len(df_mod):,}")

    df_mod['residual'] = model.resid
    df_mod['fitted']   = model.fittedvalues
    return {
        'df': df_mod, 'model': model,
        'label_base': label_base, 'label_comp': label_comp,
        'kind': 'pair',
    }


# ==========================================
# REGRESSION ENGINE — N-election pool
# ==========================================
def run_pooled_n_regression(
    dfs_by_key:    dict,        # {election_key: DataFrame}
    keys_ordered:  list,        # ordered; first = reference
    pool_label:    str,
) -> dict:
    """
    Pools N elections using C(election_id) categorical treatment variable.
    The first key in keys_ordered is the reference level.
    Shift coefficients are the C(election_id)[T.<key>]:covariate interactions.
    """
    MIN_VOTES = 50
    labels = {k: ELECTION_CONFIGS[k]['label'] for k in keys_ordered}
    ref_key = keys_ordered[0]

    frames = []
    for k in keys_ordered:
        df = dfs_by_key[k].copy()
        df = df[(df['total_votes'] > MIN_VOTES) & (df['total_dem'] > 0)]
        df['election_id'] = k
        frames.append(df)

    df_pooled = pd.concat(frames, ignore_index=True)
    df_pooled['vote_share'] = df_pooled['total_dem'] / df_pooled['total_votes']
    # Ensure reference level is first in categorical order
    df_pooled['election_id'] = pd.Categorical(
        df_pooled['election_id'],
        categories=keys_ordered,
        ordered=False,
    )

    potential_age_cols = [c for c in AGE_GENDER_COLS if c != 'pct_f_4044']
    coverage = {c: df_pooled[c].notna().mean()
                for c in potential_age_cols if c in df_pooled.columns}
    model_age_gender_cols = [c for c, cov in coverage.items() if cov >= 0.50]

    if model_age_gender_cols:
        print(f"  Census covariates: {len(model_age_gender_cols)} age/gender columns.")
    else:
        print("  [!] No census data — reduced model.")

    base_req = ['vote_share', 'election_id', 'province_tag', 'urban_type',
                'is_military', 'sum_people', 'log_apt_price']
    df_mod = df_pooled.dropna(subset=base_req).copy()
    if model_age_gender_cols:
        df_mod = df_mod.dropna(subset=model_age_gender_cols).copy()

    cont_cols = model_age_gender_cols + ['log_apt_price']
    scaler = StandardScaler()
    if cont_cols:
        df_mod[cont_cols] = scaler.fit_transform(df_mod[cont_cols])

    # Reference level set via Treatment() contrasts
    ref_str     = f"'{ref_key}'"
    elec_term   = f"C(election_id, Treatment({ref_str}))"
    age_part    = (' + '.join(model_age_gender_cols) + ' + ') if model_age_gender_cols else ''
    base_cov    = (f"{age_part}is_military + log_apt_price "
                   f"+ C(province_tag) + C(urban_type)")

    if model_age_gender_cols:
        cohort_shift = ' + '.join([f"{elec_term}:{c}" for c in model_age_gender_cols])
        formula = (f"vote_share ~ {base_cov} + {elec_term} "
                   f"+ {elec_term}:C(province_tag) + {cohort_shift}")
    else:
        formula = f"vote_share ~ {base_cov} + {elec_term} + {elec_term}:C(province_tag)"

    print(f"  Fitting N={len(keys_ordered)}-election pool: {pool_label} ...")
    model = smf.wls(formula, data=df_mod, weights=df_mod['sum_people']).fit(cov_type='HC3')
    print(f"  R²: {model.rsquared*100:.2f}%  |  N: {len(df_mod):,}")

    df_mod['residual'] = model.resid
    df_mod['fitted']   = model.fittedvalues
    return {
        'df':                df_mod,
        'model':             model,
        'pool_label':        pool_label,
        'keys_ordered':      keys_ordered,
        'ref_key':           ref_key,
        'labels':            labels,
        'model_age_cols':    model_age_gender_cols,
        'kind':              'pool',
    }


# ==========================================
# COEFFICIENT EXTRACTION HELPERS
# ==========================================
ELECTION_COLORS = {
    'gen21':  'dodgerblue',
    'gen22':  'crimson',
    'pres20': 'forestgreen',
    'pres21': 'darkorange',
}

def _clean_param_name(n: str) -> str:
    n = re.sub(r"C\(election_id,\s*Treatment\('[^']+'\)\)\[T\.", '', n)
    n = re.sub(r"C\(election_id,\s*Treatment\('[^']+'\)\)",       'election_id', n)
    n = n.replace('C(province_tag)[T.', 'Region: ')
    n = n.replace('C(urban_type)[T.',   'Urban: ')
    n = n.replace(']', '')
    return n


def _extract_pair_coefs(results: dict) -> pd.DataFrame:
    """
    For a 2-election pair result.
    Returns columns: feature, est_base, err_base, est_comp, err_comp.
    """
    model   = results['model']
    params  = model.params
    cov_mat = model.cov_params()

    base_names = [c for c in params.index
                  if ':' not in c and c not in ('Intercept', 'is_comp')]
    rows = []
    for b in base_names:
        est_b = params[b] * 100
        se_b  = np.sqrt(cov_mat.loc[b, b]) * 100
        sc    = f"is_comp:{b}"
        if sc in params:
            est_c = (est_b + params[sc] * 100)
            var_c = (cov_mat.loc[b, b] + cov_mat.loc[sc, sc]
                     + 2 * cov_mat.loc[b, sc])
            se_c  = np.sqrt(var_c) * 100
        else:
            est_c, se_c = est_b, se_b
        rows.append({'feature': b,
                     'est_base': est_b, 'err_base': 1.96 * se_b,
                     'est_comp': est_c, 'err_comp': 1.96 * se_c})
    df = pd.DataFrame(rows).set_index('feature')
    df.index = (df.index
                .str.replace('C(province_tag)[T.', 'Region: ', regex=False)
                .str.replace('C(urban_type)[T.',   'Urban: ',  regex=False)
                .str.replace(']', '', regex=False))
    return df.sort_values('est_base')


def _extract_pool_coefs(results: dict) -> dict:
    """
    For an N-election pool result.
    Returns {election_key: DataFrame(feature, est, err)} for every election,
    including the reference.  The reference uses the base coefficients directly;
    non-reference elections use base + interaction with joint SE.
    """
    model        = results['model']
    params       = model.params
    cov_mat      = model.cov_params()
    keys_ordered = results['keys_ordered']
    ref_key      = results['ref_key']

    # Base feature names = params without any election_id interaction or intercept
    elec_prefix  = "C(election_id"
    base_names   = [c for c in params.index
                    if elec_prefix not in c and c != 'Intercept']

    # Helper: find the interaction param name for a given key × base feature
    def find_interaction(key, base):
        # statsmodels may order the interaction either way
        candidates = [
            f"C(election_id, Treatment('{ref_key}'))[T.{key}]:{base}",
            f"{base}:C(election_id, Treatment('{ref_key}'))[T.{key}]",
        ]
        for c in candidates:
            if c in params: return c
        return None

    out = {}
    for key in keys_ordered:
        rows = []
        for b in base_names:
            est_b = params[b] * 100
            se_b  = np.sqrt(cov_mat.loc[b, b]) * 100

            if key == ref_key:
                rows.append({'feature': b,
                             'est': est_b, 'err': 1.96 * se_b})
            else:
                ic = find_interaction(key, b)
                if ic:
                    est_k = est_b + params[ic] * 100
                    var_k = (cov_mat.loc[b, b] + cov_mat.loc[ic, ic]
                             + 2 * cov_mat.loc[b, ic])
                    se_k  = np.sqrt(var_k) * 100
                else:
                    est_k, se_k = est_b, se_b
                rows.append({'feature': b,
                             'est': est_k, 'err': 1.96 * se_k})
        df = pd.DataFrame(rows).set_index('feature')
        df.index = (df.index
                    .str.replace('C(province_tag)[T.', 'Region: ', regex=False)
                    .str.replace('C(urban_type)[T.',   'Urban: ',  regex=False)
                    .str.replace(']', '', regex=False))
        out[key] = df
    return out


# ==========================================
# VISUALIZATION — 2-election pair dashboard
# ==========================================
def plot_pair_dashboard(results: dict, out_path: str):
    model      = results['model']
    df_mod     = results['df']
    label_base = results['label_base']
    label_comp = results['label_comp']

    df_coef = _extract_pair_coefs(results)

    fig    = plt.figure(figsize=(22, 24))
    gs_fig = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[2.5, 1, 1])
    fig.suptitle(f"Electoral Shift Dashboard\n{label_base}  →  {label_comp}",
                 fontsize=18, fontweight='bold', y=0.98)

    ax_coef = fig.add_subplot(gs_fig[0, :])
    y_pos   = np.arange(len(df_coef))
    for i in range(len(df_coef)):
        ax_coef.plot([df_coef['est_base'].iloc[i], df_coef['est_comp'].iloc[i]],
                     [y_pos[i] - 0.15, y_pos[i] + 0.15],
                     color='gray', alpha=0.3, zorder=1)
    ax_coef.errorbar(df_coef['est_base'], y_pos - 0.15, xerr=df_coef['err_base'],
                     fmt='o', color='dodgerblue', markersize=6, elinewidth=2,
                     label=label_base, zorder=2)
    ax_coef.errorbar(df_coef['est_comp'], y_pos + 0.15, xerr=df_coef['err_comp'],
                     fmt='o', color='crimson', markersize=6, elinewidth=2,
                     label=label_comp, zorder=3)
    ax_coef.axvline(0, color='black', linestyle='--', linewidth=1)
    ax_coef.set_yticks(y_pos)
    ax_coef.set_yticklabels(df_coef.index, fontsize=10)
    ax_coef.set_title('Coefficient Shift (Pooled DiD Model)', fontsize=14)
    ax_coef.set_xlabel('Effect on Dem Vote Share (Percentage Points)', fontsize=12)
    ax_coef.legend(fontsize=12, loc='lower right')

    df_b = df_mod[df_mod['is_comp'] == 0]
    df_c = df_mod[df_mod['is_comp'] == 1]

    def plot_avp(ax, df_sub, title, color):
        x, y = df_sub['fitted'] * 100, df_sub['vote_share'] * 100
        ax.scatter(x, y, alpha=0.3, s=15, color=color, edgecolors='none')
        mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, alpha=0.6)
        ax.set_title(f'Actual vs Predicted — {title}', fontsize=12)
        ax.set_xlabel('Predicted (%)'); ax.set_ylabel('Actual (%)')

    def plot_res(ax, df_sub, title, color):
        res = df_sub['residual'] * 100
        ax.hist(res, bins=45, density=True, color=color, alpha=0.6)
        mu, std = res.mean(), res.std()
        x_line  = np.linspace(*ax.get_xlim(), 100)
        ax.plot(x_line, norm.pdf(x_line, mu, std), 'k', lw=2)
        ax.axvline(0, color='red', lw=1, ls='--', alpha=0.5)
        ax.set_title(f'Residuals — {title}  (μ={mu:.2f}, σ={std:.2f})', fontsize=12)
        ax.set_xlabel('Residual Error (%)')

    plot_avp(fig.add_subplot(gs_fig[1, 0]), df_b, label_base, 'dodgerblue')
    plot_avp(fig.add_subplot(gs_fig[1, 1]), df_c, label_comp, 'crimson')
    plot_res(fig.add_subplot(gs_fig[2, 0]), df_b, label_base, 'dodgerblue')
    plot_res(fig.add_subplot(gs_fig[2, 1]), df_c, label_comp, 'crimson')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=180); plt.close()
    print(f"  → Saved dashboard: {out_path}")
    report_path = out_path.replace('.png', '_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"=== {label_base}  →  {label_comp} ===\n\n")
        f.write(model.summary().as_text())
    print(f"  → Saved report:    {report_path}")


# ==========================================
# VISUALIZATION — N-election pool dashboard
# ==========================================
def plot_pool_dashboard(results: dict, out_path: str):
    """
    Dashboard for an N-election pooled model.
    Top panel: dumbbell with one dot per election per covariate.
    Bottom: actual-vs-predicted + residual histogram per election.
    """
    df_mod       = results['df']
    keys_ordered = results['keys_ordered']
    labels       = results['labels']
    pool_label   = results['pool_label']
    n_elec       = len(keys_ordered)

    coef_by_key = _extract_pool_coefs(results)

    # Use reference election to set feature order
    ref_key    = results['ref_key']
    ref_df     = coef_by_key[ref_key].sort_values('est')
    feat_order = ref_df.index.tolist()

    # Align all elections to same feature order
    for k in keys_ordered:
        coef_by_key[k] = coef_by_key[k].reindex(feat_order)

    n_feat  = len(feat_order)
    y_pos   = np.arange(n_feat)
    offsets = np.linspace(-0.2, 0.2, n_elec)

    n_bottom_cols = min(n_elec, 4)   # at most 4 columns for avp/residual rows
    fig = plt.figure(figsize=(24, 6 + n_feat * 0.35 + 10))
    gs_fig = gridspec.GridSpec(
        3, n_bottom_cols, figure=fig,
        height_ratios=[max(n_feat * 0.35, 8), 4, 4],
    )
    fig.suptitle(f"Pool Dashboard\n{pool_label}", fontsize=16, fontweight='bold', y=0.99)

    # ── Dumbbell ──────────────────────────────────────────────────────────────
    ax_coef = fig.add_subplot(gs_fig[0, :])
    for i, (k, offset) in enumerate(zip(keys_ordered, offsets)):
        df_k  = coef_by_key[k]
        color = ELECTION_COLORS.get(k, f'C{i}')
        # connecting lines between reference and this election
        if k != ref_key:
            for fi, feat in enumerate(feat_order):
                ax_coef.plot(
                    [coef_by_key[ref_key].loc[feat, 'est'], df_k.loc[feat, 'est']],
                    [y_pos[fi] + offsets[0], y_pos[fi] + offset],
                    color='gray', alpha=0.2, zorder=1,
                )
        ax_coef.errorbar(
            df_k['est'], y_pos + offset, xerr=df_k['err'],
            fmt='o', color=color, markersize=5, elinewidth=1.8,
            label=labels[k], zorder=2 + i,
        )
    ax_coef.axvline(0, color='black', linestyle='--', linewidth=1)
    ax_coef.set_yticks(y_pos)
    ax_coef.set_yticklabels(feat_order, fontsize=9)
    ax_coef.set_title('Derived Coefficients per Election (Reference = ' + labels[ref_key] + ')',
                       fontsize=13)
    ax_coef.set_xlabel('Effect on Dem Vote Share (Percentage Points)', fontsize=11)
    ax_coef.legend(fontsize=11, loc='lower right')

    # ── Per-election actual-vs-predicted & residuals ─────────────────────────
    def plot_avp(ax, df_sub, title, color):
        x, y = df_sub['fitted'] * 100, df_sub['vote_share'] * 100
        if len(x) == 0: return
        ax.scatter(x, y, alpha=0.3, s=12, color=color, edgecolors='none')
        mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1.2, alpha=0.6)
        ax.set_title(f'AvP — {title}', fontsize=9)
        ax.set_xlabel('Pred (%)'); ax.set_ylabel('Actual (%)')
        ax.tick_params(labelsize=8)

    def plot_res(ax, df_sub, title, color):
        res = df_sub['residual'] * 100
        if len(res) == 0: return
        ax.hist(res, bins=40, density=True, color=color, alpha=0.6)
        mu, std = res.mean(), res.std()
        x_line  = np.linspace(*ax.get_xlim(), 100)
        ax.plot(x_line, norm.pdf(x_line, mu, std), 'k', lw=1.5)
        ax.axvline(0, color='red', lw=1, ls='--', alpha=0.5)
        ax.set_title(f'Resid — {title}\n(μ={mu:.2f}, σ={std:.2f})', fontsize=9)
        ax.set_xlabel('Error (%)')
        ax.tick_params(labelsize=8)

    for col_i, k in enumerate(keys_ordered[:n_bottom_cols]):
        df_k  = df_mod[df_mod['election_id'] == k]
        color = ELECTION_COLORS.get(k, f'C{col_i}')
        plot_avp(fig.add_subplot(gs_fig[1, col_i]), df_k, labels[k], color)
        plot_res(fig.add_subplot(gs_fig[2, col_i]), df_k, labels[k], color)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path, dpi=180, bbox_inches='tight'); plt.close()
    print(f"  → Saved pool dashboard: {out_path}")

    report_path = out_path.replace('.png', '_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"=== {pool_label} ===\n\n")
        f.write(results['model'].summary().as_text())
    print(f"  → Saved report:         {report_path}")


# ==========================================
# SUMMARY SHIFT OVERVIEW — pair results
# ==========================================
def plot_summary_overview(all_results: list, out_path: str):
    """
    One column per pair result showing is_comp interaction shift coefficients.
    """
    pair_results = [r for r in all_results if r.get('kind') == 'pair']
    if not pair_results:
        return
    n = len(pair_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 14), sharey=False)
    if n == 1: axes = [axes]
    fig.suptitle(
        "Dem Vote-Share Shift Coefficients — Pairwise Comparisons\n"
        "(is_comp × covariate interactions, pp, 95 % CI)",
        fontsize=15, fontweight='bold', y=1.01,
    )
    for ax, res in zip(axes, pair_results):
        model  = res['model']
        params = model.params
        cov_m  = model.cov_params()
        shift_rows = [
            (
                _clean_param_name(n).replace('is_comp:', ''),
                params[n] * 100,
                np.sqrt(cov_m.loc[n, n]) * 100,
            )
            for n in params.index if n.startswith('is_comp:')
        ]
        if not shift_rows:
            ax.text(0.5, 0.5, 'No interactions', ha='center', va='center',
                    transform=ax.transAxes)
            ax.set_title(f"{res['label_base']}\n→ {res['label_comp']}", fontsize=9)
            continue
        labels_s = [r[0] for r in shift_rows]
        ests     = np.array([r[1] for r in shift_rows])
        errs     = np.array([r[2] * 1.96 for r in shift_rows])
        order    = np.argsort(ests)
        labels_s = [labels_s[i] for i in order]
        ests, errs = ests[order], errs[order]
        colors   = ['crimson' if e > 0 else 'dodgerblue' for e in ests]
        ax.barh(np.arange(len(labels_s)), ests, xerr=errs, color=colors,
                alpha=0.75, error_kw={'elinewidth': 1.2, 'capsize': 2})
        ax.axvline(0, color='black', lw=1, ls='--')
        ax.set_yticks(np.arange(len(labels_s)))
        ax.set_yticklabels(labels_s, fontsize=8)
        ax.set_xlabel('Shift (pp)', fontsize=9)
        ax.set_title(
            f"{res['label_base']}\n→ {res['label_comp']}\n"
            f"(R²={res['model'].rsquared*100:.1f}%)", fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', labelsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight'); plt.close()
    print(f"  → Saved pair overview: {out_path}")


# ==========================================
# REGRESSION ENGINE — single election
# ==========================================
def run_single_regression(df: pd.DataFrame, election_key: str) -> dict:
    """
    Cross-sectional WLS for one election.
    No election_id terms — returns absolute coefficients for all covariates.
    """
    MIN_VOTES = 50
    cfg   = ELECTION_CONFIGS[election_key]
    label = cfg['label']

    df = df[(df['total_votes'] > MIN_VOTES) & (df['total_dem'] > 0)].copy()
    df['vote_share'] = df['total_dem'] / df['total_votes']

    potential_age_cols = [c for c in AGE_GENDER_COLS if c != 'pct_f_4044']
    coverage = {c: df[c].notna().mean()
                for c in potential_age_cols if c in df.columns}
    model_age_cols = [c for c, cov in coverage.items() if cov >= 0.50]

    if model_age_cols:
        print(f"  Census covariates: {len(model_age_cols)} age/gender columns.")
    else:
        print("  [!] No census data — reduced model.")

    base_req = ['vote_share', 'province_tag', 'urban_type',
                'is_military', 'sum_people', 'log_apt_price']
    df_mod = df.dropna(subset=base_req).copy()
    if model_age_cols:
        df_mod = df_mod.dropna(subset=model_age_cols).copy()

    cont_cols = model_age_cols + ['log_apt_price']
    scaler = StandardScaler()
    if cont_cols:
        df_mod[cont_cols] = scaler.fit_transform(df_mod[cont_cols])

    age_part = (' + '.join(model_age_cols) + ' + ') if model_age_cols else ''
    formula  = (f"vote_share ~ {age_part}is_military + log_apt_price "
                f"+ C(province_tag) + C(urban_type)")

    print(f"  Fitting single-election model: {label} ...")
    model = smf.wls(formula, data=df_mod, weights=df_mod['sum_people']).fit(cov_type='HC3')
    print(f"  R²: {model.rsquared*100:.2f}%  |  N: {len(df_mod):,}")

    df_mod['residual'] = model.resid
    df_mod['fitted']   = model.fittedvalues
    return {
        'df':              df_mod,
        'model':           model,
        'label':           label,
        'election_key':    election_key,
        'model_age_cols':  model_age_cols,
        'kind':            'single',
    }


# ==========================================
# VISUALIZATION — single election dashboard
# ==========================================
def plot_single_dashboard(results: dict, out_path: str):
    """
    Three-panel dashboard for one election:
      Left col  : horizontal bar chart of all coefficients with 95% CI
      Right top : actual vs predicted scatter
      Right bot : residual histogram
    """
    model = results['model']
    df_mod = results['df']
    label  = results['label']
    color  = ELECTION_COLORS.get(results['election_key'], 'steelblue')

    params  = model.params
    cov_mat = model.cov_params()

    # Build coefficient table (exclude Intercept)
    rows = []
    for name in params.index:
        if name == 'Intercept':
            continue
        est = params[name] * 100
        se  = np.sqrt(cov_mat.loc[name, name]) * 100
        pv  = model.pvalues[name]
        clean = (name
                 .replace('C(province_tag)[T.', 'Region: ')
                 .replace('C(urban_type)[T.',   'Urban: ')
                 .replace(']', ''))
        rows.append({'feature': clean, 'est': est,
                     'err': 1.96 * se, 'pvalue': pv})

    df_coef = (pd.DataFrame(rows)
               .set_index('feature')
               .sort_values('est'))

    # Significance marker
    def sig_marker(p):
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        return ''

    fig = plt.figure(figsize=(22, max(12, len(df_coef) * 0.38 + 4)))
    gs_fig = gridspec.GridSpec(2, 2, figure=fig,
                               width_ratios=[1.6, 1],
                               height_ratios=[1, 1])
    fig.suptitle(
        f"Single-Election Analysis: {label}\n"
        f"Cross-sectional WLS — Dem Vote Share  "
        f"(R\u00b2={model.rsquared*100:.1f}%,  N={len(df_mod):,})",
        fontsize=16, fontweight='bold', y=0.99,
    )

    # ── Coefficient bar chart ─────────────────────────────────────────────────
    ax_coef = fig.add_subplot(gs_fig[:, 0])   # spans both rows on left
    y_pos   = np.arange(len(df_coef))
    bar_colors = [color if e > 0 else 'lightcoral'
                  if color != 'lightcoral' else 'dodgerblue'
                  for e in df_coef['est']]
    ax_coef.barh(y_pos, df_coef['est'], xerr=df_coef['err'],
                 color=bar_colors, alpha=0.80,
                 error_kw={'elinewidth': 1.4, 'capsize': 3})
    # Significance labels
    x_max = (df_coef['est'] + df_coef['err']).max()
    for yi, (feat, row) in enumerate(df_coef.iterrows()):
        mark = sig_marker(row['pvalue'])
        if mark:
            ax_coef.text(x_max * 1.02, yi, mark,
                         va='center', fontsize=9, color='black')
    ax_coef.axvline(0, color='black', linestyle='--', linewidth=1)
    ax_coef.set_yticks(y_pos)
    ax_coef.set_yticklabels(df_coef.index, fontsize=9)
    ax_coef.set_xlabel('Effect on Dem Vote Share (Percentage Points)', fontsize=11)
    ax_coef.set_title(
        'Coefficient Estimates  (* p<.05  ** p<.01  *** p<.001)',
        fontsize=12,
    )

    # ── Actual vs Predicted ───────────────────────────────────────────────────
    ax_avp = fig.add_subplot(gs_fig[0, 1])
    x = df_mod['fitted']     * 100
    y = df_mod['vote_share'] * 100
    ax_avp.scatter(x, y, alpha=0.35, s=14, color=color, edgecolors='none')
    mn, mx = min(x.min(), y.min()), max(x.max(), y.max())
    ax_avp.plot([mn, mx], [mn, mx], 'k--', lw=1.5, alpha=0.6)
    ax_avp.set_title('Actual vs Predicted', fontsize=12)
    ax_avp.set_xlabel('Predicted Vote Share (%)', fontsize=10)
    ax_avp.set_ylabel('Actual Vote Share (%)',    fontsize=10)

    # ── Residual histogram ────────────────────────────────────────────────────
    ax_res = fig.add_subplot(gs_fig[1, 1])
    res    = df_mod['residual'] * 100
    ax_res.hist(res, bins=50, density=True, color=color, alpha=0.65)
    mu, std = res.mean(), res.std()
    x_line  = np.linspace(*ax_res.get_xlim(), 200)
    ax_res.plot(x_line, norm.pdf(x_line, mu, std), 'k', lw=2)
    ax_res.axvline(0, color='red', lw=1, ls='--', alpha=0.6)
    ax_res.set_title(f'Residuals  (μ={mu:.2f} pp,  σ={std:.2f} pp)', fontsize=12)
    ax_res.set_xlabel('Residual Error (pp)', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  → Saved single dashboard: {out_path}")

    report_path = out_path.replace('.png', '_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"=== Single-Election Analysis: {label} ===\n\n")
        f.write(model.summary().as_text())
    print(f"  → Saved report:            {report_path}")


# ==========================================
# SUMMARY — all single elections side-by-side
# ==========================================
def plot_single_overview(all_single: list, out_path: str):
    """
    One column per election, showing every coefficient bar chart
    on a shared y-axis so magnitudes are directly comparable.
    """
    n = len(all_single)
    if n == 0:
        return

    # Collect union of feature names and a common sort order
    # (use the first election's ordering as the reference)
    ref_params = all_single[0]['model'].params
    all_features = []
    for res in all_single:
        for name in res['model'].params.index:
            if name == 'Intercept': continue
            clean = (name
                     .replace('C(province_tag)[T.', 'Region: ')
                     .replace('C(urban_type)[T.',   'Urban: ')
                     .replace(']', ''))
            if clean not in all_features:
                all_features.append(clean)

    # Sort by mean absolute estimate across elections
    feat_mean_abs = {}
    for feat in all_features:
        vals = []
        for res in all_single:
            params = res['model'].params
            cov_m  = res['model'].cov_params()
            # reverse-map clean name → raw param name
            for raw in params.index:
                clean = (raw
                         .replace('C(province_tag)[T.', 'Region: ')
                         .replace('C(urban_type)[T.',   'Urban: ')
                         .replace(']', ''))
                if clean == feat:
                    vals.append(abs(params[raw] * 100))
                    break
        feat_mean_abs[feat] = np.mean(vals) if vals else 0
    ordered_feats = sorted(all_features, key=lambda f: feat_mean_abs[f])

    y_pos  = np.arange(len(ordered_feats))
    fig, axes = plt.subplots(1, n, figsize=(7 * n, max(10, len(ordered_feats) * 0.3 + 3)),
                             sharey=True)
    if n == 1: axes = [axes]

    fig.suptitle(
        "Single-Election Coefficient Comparison (all elections, shared scale)\n"
        "Effect on Dem Vote Share (pp, 95% CI)  —  * p<.05  ** p<.01  *** p<.001",
        fontsize=14, fontweight='bold', y=1.01,
    )

    for ax, res in zip(axes, all_single):
        key    = res['election_key']
        color  = ELECTION_COLORS.get(key, 'steelblue')
        params = res['model'].params
        cov_m  = res['model'].cov_params()
        pvals  = res['model'].pvalues

        ests, errs, colors_bar, marks = [], [], [], []
        for feat in ordered_feats:
            est_val = err_val = 0.0
            pv = 1.0
            for raw in params.index:
                clean = (raw
                         .replace('C(province_tag)[T.', 'Region: ')
                         .replace('C(urban_type)[T.',   'Urban: ')
                         .replace(']', ''))
                if clean == feat and raw != 'Intercept':
                    est_val = params[raw] * 100
                    err_val = 1.96 * np.sqrt(cov_m.loc[raw, raw]) * 100
                    pv      = pvals[raw]
                    break
            ests.append(est_val)
            errs.append(err_val)
            colors_bar.append(color if est_val >= 0 else 'lightgray')
            marks.append('***' if pv < 0.001 else '**' if pv < 0.01
                         else '*' if pv < 0.05 else '')

        ests = np.array(ests)
        errs = np.array(errs)
        ax.barh(y_pos, ests, xerr=errs, color=colors_bar, alpha=0.78,
                error_kw={'elinewidth': 1.1, 'capsize': 2})
        x_max = (ests + errs).max() if len(ests) else 1
        for yi, mark in enumerate(marks):
            if mark:
                ax.text(x_max * 1.03, yi, mark, va='center', fontsize=8)
        ax.axvline(0, color='black', lw=1, ls='--')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(ordered_feats, fontsize=8)
        ax.set_xlabel('Effect (pp)', fontsize=9)
        ax.set_title(
            f"{res['label']}\n(R²={res['model'].rsquared*100:.1f}%,  N={len(res['df']):,})",
            fontsize=10, fontweight='bold',
        )
        ax.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"  → Saved single overview:   {out_path}")


# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"\n{'='*65}")
    print("  Korean Election Shift Analysis — General & Presidential")
    print(f"{'='*65}")

    # ----------------------------------------------------------
    # STEP 1: Load census + apt covariates
    # ----------------------------------------------------------
    census_cache = {}
    apt_cache    = {}

    for key in ('gen21', 'gen22'):
        cfg = ELECTION_CONFIGS[key]
        print(f"\n[Census/Apt] Loading covariates for {cfg['label']} ...")
        cen = load_census_csv(cfg.get('census_csv', '__missing__'))
        apt = load_apt_csv(cfg.get('apt_csv_glob', '__missing__'))
        census_cache[key] = cen
        apt_cache[key]    = apt
        print(f"  Census rows: {len(cen)}  |  Apt rows: {len(apt)}")

    for key in ('pres20', 'pres21'):
        cfg      = ELECTION_CONFIGS[key]
        fallback = cfg['census_key']
        print(f"\n[Census/Apt] {cfg['label']} → checking dedicated files ...")
        cen = load_census_csv(cfg.get('census_csv', '__missing__'))
        apt = load_apt_csv(cfg.get('apt_csv_glob', '__missing__'))
        if cen.empty:
            cen = census_cache[fallback]
            print(f"  No dedicated census → borrowing from {ELECTION_CONFIGS[fallback]['label']}")
        if apt.empty:
            apt = apt_cache[fallback]
            print(f"  No dedicated apt data → borrowing from {ELECTION_CONFIGS[fallback]['label']}")
        census_cache[key] = cen
        apt_cache[key]    = apt

    # ----------------------------------------------------------
    # STEP 2: Load & merge each election
    # ----------------------------------------------------------
    datasets = {}
    for key, cfg in ELECTION_CONFIGS.items():
        print(f"\n[Election] Loading {cfg['label']} ...")
        df_elec = load_election_csv(cfg)
        if df_elec.empty:
            print(f"  [!] No data loaded for {key}. Skipping.")
            continue
        cen_key = cfg['census_key']
        df_merged = merge_dong_with_covariates(
            df_elec,
            census_cache.get(cen_key, pd.DataFrame()),
            apt_cache.get(cen_key, pd.DataFrame()),
        )
        datasets[key] = df_merged
        print(f"  Merged {len(df_merged):,} dong-level observations.")

    # ----------------------------------------------------------
    # STEP 3: Pairwise comparisons (binary DiD)
    # ----------------------------------------------------------
    print(f"\n{'='*65}")
    print("  PART A — Pairwise Comparisons")
    print(f"{'='*65}")
    all_results = []
    for base_key, comp_key, pair_label, series in COMPARISON_PAIRS:
        if base_key not in datasets or comp_key not in datasets:
            print(f"\n[!] Skipping '{pair_label}' — missing data.")
            continue
        print(f"\n{'─'*65}\n  {pair_label}\n{'─'*65}")
        res = run_pooled_shift_regression(
            datasets[base_key], datasets[comp_key],
            ELECTION_CONFIGS[base_key]['label'],
            ELECTION_CONFIGS[comp_key]['label'],
        )
        res['pair_label'] = pair_label
        res['series']     = series
        all_results.append(res)
        safe = (pair_label
                .replace(' ', '_').replace(':', '').replace('(', '')
                .replace(')', '').replace('→', 'to')[:60])
        plot_pair_dashboard(res, out_path=f"dashboard_{safe}.png")

    if all_results:
        plot_summary_overview(all_results, out_path='shift_overview_pairwise.png')

    # ----------------------------------------------------------
    # STEP 4: N-election pool analyses
    # ----------------------------------------------------------
    print(f"\n{'='*65}")
    print("  PART B — N-Election Pool Analyses")
    print(f"{'='*65}")
    for pool_cfg in POOL_CONFIGS:
        keys   = pool_cfg['keys']
        label  = pool_cfg['label']
        prefix = pool_cfg['out_prefix']

        missing = [k for k in keys if k not in datasets]
        if missing:
            print(f"\n[!] Skipping pool '{label}' — missing: {missing}")
            continue

        print(f"\n{'─'*65}\n  Pool: {label}\n{'─'*65}")
        dfs = {k: datasets[k] for k in keys}
        res = run_pooled_n_regression(dfs, keys, label)
        plot_pool_dashboard(res, out_path=f"{prefix}_dashboard.png")

    # ----------------------------------------------------------
    # STEP 5: Single-election analyses
    # ----------------------------------------------------------
    print(f"\n{'='*65}")
    print("  PART C — Single-Election Analyses")
    print(f"{'='*65}")
    single_results = []
    for scfg in SINGLE_CONFIGS:
        key    = scfg['key']
        prefix = scfg['out_prefix']
        if key not in datasets:
            print(f"\n[!] Skipping single '{ELECTION_CONFIGS[key]['label']}' — no data.")
            continue
        print(f"\n{'─'*65}\n  Single: {ELECTION_CONFIGS[key]['label']}\n{'─'*65}")
        res = run_single_regression(datasets[key], key)
        single_results.append(res)
        plot_single_dashboard(res, out_path=f"{prefix}_dashboard.png")

    if len(single_results) > 1:
        plot_single_overview(single_results, out_path='single_overview_all.png')

    print(f"\n{'='*65}\n  Done.\n{'='*65}\n")

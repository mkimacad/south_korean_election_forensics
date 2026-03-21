import pandas as pd
import numpy as np
import math
import os
import re
import glob
from difflib import get_close_matches
from scipy.stats import norm, probplot
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

# ==========================================
# MATPLOTLIB & KOREAN FONT SETUP
# ==========================================
import matplotlib.pyplot as plt
try:
    import koreanize_matplotlib
except ImportError:
    print("[!] koreanize_matplotlib not found. Run: pip install koreanize-matplotlib")

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
        'dashboard_out': 'election_regression_dashboard_21st.png',
    },
    22: {
        'census_csv':    '22nd_election_census.csv',
        'result_csv':    '22nd_election_result.csv',
        'apt_csv_glob':  '*22nd_election_*_apt_price.csv', 
        'dem_pattern':   r'더불어민주당',
        'con_pattern':   r'국민의힘',
        'label':         '22nd General Election (2024)',
        'dashboard_out': 'election_regression_dashboard_22nd.png',
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

# The 22 dynamic age-gender columns we will track
AGE_GENDER_COLS = [
    'pct_m_1824', 'pct_m_2529', 'pct_m_3034', 'pct_m_3539', 'pct_m_4044', 'pct_m_4549', 
    'pct_m_5054', 'pct_m_5559', 'pct_m_6064', 'pct_m_6569', 'pct_m_70plus',
    'pct_f_1824', 'pct_f_2529', 'pct_f_3034', 'pct_f_3539', 'pct_f_4044', 'pct_f_4549', 
    'pct_f_5054', 'pct_f_5559', 'pct_f_6064', 'pct_f_6569', 'pct_f_70plus'
]

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
    print(f"\n--- [1/4] Loading Demographic Census Data (Age-Gender cohorts) ---")
    if not os.path.exists(csv_path): return pd.DataFrame()
    try:
        try: df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError: df = pd.read_csv(csv_path, encoding='cp949', low_memory=False)

        prefix = _detect_year_prefix(df)
        
        # Total voting population (18+)
        voting_age_cols = ([f"{prefix}_계_{a}세" for a in range(18, 100)] + [f"{prefix}_계_100세 이상"])
        all_target_cols = list(voting_age_cols)
        for g in ['남', '여']:
            for a in range(18, 100):
                all_target_cols.append(f"{prefix}_{g}_{a}세")
            all_target_cols.append(f"{prefix}_{g}_100세 이상")
        
        for col in set(all_target_cols):
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False).pipe(pd.to_numeric, errors='coerce').fillna(0)

        df = df.copy() 
        df['total_voting_pop'] = df[[c for c in voting_age_cols if c in df.columns]].sum(axis=1)
        df = df[df['total_voting_pop'] > 0].copy()

        # Extract 5-year Age-Gender Cohorts dynamically
        ranges = [(18, 25, '1824'), (25, 30, '2529'), (30, 35, '3034'), (35, 40, '3539'),
                  (40, 45, '4044'), (45, 50, '4549'), (50, 55, '5054'), (55, 60, '5559'),
                  (60, 65, '6064'), (65, 70, '6569')]
        
        for g, g_str in [('남', 'm'), ('여', 'f')]:
            for r_start, r_end, r_str in ranges:
                cols = [f"{prefix}_{g}_{a}세" for a in range(r_start, r_end)]
                col_name = f'pct_{g_str}_{r_str}'
                df[col_name] = df[[c for c in cols if c in df.columns]].sum(axis=1) / df['total_voting_pop']
            
            # 70+ logic
            cols_70 = [f"{prefix}_{g}_{a}세" for a in range(70, 100)] + [f"{prefix}_{g}_100세 이상"]
            col_name_70 = f'pct_{g_str}_70plus'
            df[col_name_70] = df[[c for c in cols_70 if c in df.columns]].sum(axis=1) / df['total_voting_pop']

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
                'sgg_candidates': sgg_cands, 'primary_sgg': sgg_cands[0] if sgg_cands else "",
                'dong_norm': dong_norm, 'dong_raw': row['행정구역']
            }
            # Append all 22 age-gender cohorts
            for col in AGE_GENDER_COLS:
                row_dict[col] = row.get(col, np.nan)
            rows.append(row_dict)

        census = pd.DataFrame(rows)
        print(f"    Loaded {len(census):,} census rows with 22 Age-Gender cohorts.")
        return census
    except Exception as e:
        print(f"[!] Error processing census CSV: {e}")
        return pd.DataFrame()

def load_apt_csv(glob_pattern: str) -> pd.DataFrame:
    print(f"\n--- [2/4] Loading Apartment Transaction Data ({glob_pattern}) ---")
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
    print(f"\n--- [3/4] Loading Election Result Data ({csv_path}) ---")
    try:
        try: df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError: df = pd.read_csv(csv_path, encoding='cp949', low_memory=False)
    except Exception as e:
        print(f"[!] Failed to read election CSV: {e}")
        return pd.DataFrame(), pd.DataFrame()

    df['득표수']       = pd.to_numeric(df['득표수'], errors='coerce').fillna(0).astype(int)
    df['is_dem']      = df['후보자'].str.contains(dem_pattern, case=False, na=False)
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

    # Aggregate Votes at Dong Level
    total_dem = df_votes[df_votes['is_dem']].groupby(dong_key)['득표수'].sum().reset_index(name='total_dem')
    total_votes = df_votes.groupby(dong_key)['득표수'].sum().reset_index(name='total_votes')
    sum_people_dong = df_geo[df_geo['후보자'] == '선거인수'].groupby(dong_key)['득표수'].sum().reset_index(name='sum_people')

    df_dong = total_dem.copy()
    for frame in (total_votes, sum_people_dong):
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

    # Aggregate Votes at Constituency Level
    total_dem_c = df_votes[df_votes['is_dem']].groupby(const_key)['득표수'].sum().reset_index(name='total_dem')
    total_votes_c = df_votes.groupby(const_key)['득표수'].sum().reset_index(name='total_votes')
    sum_people_c = df_geo[df_geo['후보자'] == '선거인수'].groupby(const_key)['득표수'].sum().reset_index(name='sum_people')

    df_const = sum_people_c.copy()
    for frame in (total_dem_c, total_votes_c):
        df_const = df_const.merge(frame, on=const_key, how='left')
    df_const = df_const.fillna(0)

    df_const['sgg_candidates'] = df_const['선거구명'].apply(sgg_cands_from_constituency)
    df_const['primary_sgg']    = df_const['sgg_candidates'].apply(lambda x: x[0] if x else "")
    df_const['province_tag']   = df_const['시도명'].map(PROV_FULL_TO_SHORT).fillna(df_const['시도명'])

    return df_dong, df_const


# ==========================================
# 3. MULTI-PASS DONG↔CENSUS↔APT MATCHER
# ==========================================

def merge_dong_with_covariates(df_election: pd.DataFrame, df_census: pd.DataFrame, df_apt: pd.DataFrame) -> pd.DataFrame:
    if not df_census.empty:
        census_lookup = {}; census_by_sgg = {}
        for _, row in df_census.iterrows():
            dnorm = row['dong_norm']
            covs = {}
            for c in AGE_GENDER_COLS:
                covs[c] = row.get(c, np.nan)
                
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
                covs = {k: np.nan for k in AGE_GENDER_COLS}

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

    return df_out[df_out['pct_f_4044'].notna()].copy()

def merge_const_with_covariates(df_const: pd.DataFrame, df_dong_merged: pd.DataFrame) -> pd.DataFrame:
    if df_dong_merged.empty or 'pct_f_4044' not in df_dong_merged.columns:
        return df_const.copy()

    const_key = ['시도명', '선거구명']
    dm = df_dong_merged.dropna(subset=['pct_f_4044']).copy()
    
    agg_funcs = {'_reg': ('sum_people', 'sum')}
    for col in AGE_GENDER_COLS:
        dm[f'_pw_{col}'] = dm[col] * dm['sum_people']
        agg_funcs[f'_pw_{col}_sum'] = (f'_pw_{col}', 'sum')

    out_cols = list(AGE_GENDER_COLS)
    
    if 'log_apt_price' in dm.columns:
        dm['_pw_log_apt_price'] = dm['log_apt_price'] * dm['sum_people']
        agg_funcs['_pw_log_apt_price_sum'] = ('_pw_log_apt_price', 'sum')
        
    agg = dm.groupby(const_key).agg(**agg_funcs).reset_index()
    
    for col in AGE_GENDER_COLS:
        agg[col] = agg[f'_pw_{col}_sum'] / agg['_reg'].replace(0, np.nan)

    if 'log_apt_price' in dm.columns:
        agg['log_apt_price'] = agg['_pw_log_apt_price_sum'] / agg['_reg'].replace(0, np.nan)
        out_cols.append('log_apt_price')

    return df_const.merge(agg[const_key + out_cols], on=const_key, how='left')


# ==========================================
# 4. REGRESSION ENGINE 
# ==========================================

def run_vote_share_regression(df_dong_raw: pd.DataFrame, df_const_raw: pd.DataFrame, df_census: pd.DataFrame, df_apt: pd.DataFrame) -> dict:
    logs = []
    def log(msg):
        print(msg)
        logs.append(msg)

    log(f"\n--- [4/4] Electoral Regression Analysis ---")
    MIN_VOTES = 50

    # Prepare Dong Level
    dm = merge_dong_with_covariates(df_dong_raw, df_census, df_apt)
    dm = dm[(dm['total_votes'] > MIN_VOTES)].copy()
    
    # DROP ZERO-CANDIDATE DISTRICTS: Prevents the structural zero anomaly at the bottom of the scatterplot
    dm = dm[dm['total_dem'] > 0].copy()
    
    dm['vote_share'] = dm['total_dem'] / dm['total_votes']
    dm['turnout']    = dm['total_votes'] / dm['sum_people'].replace(0, np.nan)

    # Prepare Constituency Level
    cm = merge_const_with_covariates(df_const_raw, dm)
    cm = cm[(cm['total_votes'] > MIN_VOTES)].copy()
    
    # DROP ZERO-CANDIDATE DISTRICTS: Applied to constituency level as well
    cm = cm[cm['total_dem'] > 0].copy()
    
    cm['vote_share'] = cm['total_dem'] / cm['total_votes']
    cm['turnout']    = cm['total_votes'] / cm['sum_people'].replace(0, np.nan)

    log("\n" + "="*55)
    log("  LEVEL C | WLS Regression on Vote Share")
    log("="*55)

    # Use all age-gender cohorts EXCEPT pct_f_4044 to avoid dummy variable trap
    model_age_gender_cols = [c for c in AGE_GENDER_COLS if c != 'pct_f_4044']
    
    req_cols = ['vote_share', 'province_tag', 'urban_type', 'is_military', 'sum_people', 'log_apt_price'] + AGE_GENDER_COLS
    df_mod = dm.dropna(subset=req_cols).copy()
    
    scaler = StandardScaler()
    cont_cols = model_age_gender_cols + ['log_apt_price']
    df_mod[cont_cols] = scaler.fit_transform(df_mod[cont_cols])
    
    age_gender_formula_str = " + ".join(model_age_gender_cols)
    
    # Primary Regression: Dem Vote Share ~ Covariates
    formula_vote = f'vote_share ~ {age_gender_formula_str} + is_military + log_apt_price + C(province_tag) + C(urban_type)'
    model_vote = smf.wls(formula_vote, data=df_mod, weights=df_mod['sum_people']).fit(cov_type='HC3')
    
    log("\n[Regression] Demographic Model on Democratic Vote Share")
    r_squared = model_vote.rsquared
    log(f"  Demographic Explained Variance R²: {r_squared*100:.2f}%")

    df_mod['residual'] = model_vote.resid
    df_mod['fitted']   = model_vote.fittedvalues

    return {
        'dong': {'df': df_mod},
        'const': {'df': cm},
        'model_vote': model_vote,
        'logs': logs
    }

# ==========================================
# 5. DASHBOARDS & REPORTS
# ==========================================

def plot_regression_dashboard(results: dict, out_path: str, title: str):
    print(f"\nGenerating visual regression dashboard → {out_path}")
    dm = results['dong']['df']
    model = results['model_vote']
    
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Electoral Regression Dashboard  –  {title}", fontsize=16, fontweight='bold', y=0.98)

    # 1. Coefficient Plot
    ax1 = plt.subplot(1, 2, 1)
    params = model.params.drop('Intercept', errors='ignore')
    conf = model.conf_int().drop('Intercept', errors='ignore')
    errors = params - conf[0]
    
    # Sort for better visualization (split age/gender vs categorical)
    age_params = params[params.index.str.contains('pct_')]
    other_params = params[~params.index.str.contains('pct_')]
    sorted_params = pd.concat([age_params.sort_values(), other_params.sort_values()])
    sorted_errors = errors[sorted_params.index]

    ax1.errorbar(sorted_params.values, range(len(sorted_params)), xerr=sorted_errors.values, 
                 fmt='o', color='royalblue', ecolor='lightgray', elinewidth=3, capsize=0)
    ax1.axvline(0, color='red', linestyle='--', linewidth=1)
    ax1.set_yticks(range(len(sorted_params)))
    ax1.set_yticklabels(sorted_params.index, fontsize=9)
    ax1.set_title('Regression Coefficients (Effect on Dem Vote Share)', fontsize=12)
    ax1.set_xlabel('Standardized Coefficient', fontsize=10)

    # 2. Actual vs Fitted
    ax2 = plt.subplot(2, 2, 2)
    x = dm['fitted'] * 100
    y = dm['vote_share'] * 100
    ax2.scatter(x, y, alpha=0.3, s=15, color='darkmagenta', edgecolors='none')
    
    mn = min(x.min(), y.min())
    mx = max(x.max(), y.max())
    ax2.plot([mn, mx], [mn, mx], 'r--', lw=1.5, label='Perfect Prediction (1:1)')
    ax2.set_title(f'Actual vs. Predicted Vote Share (R²={model.rsquared:.3f})', fontsize=12)
    ax2.set_xlabel('Predicted Vote Share (%)', fontsize=10)
    ax2.set_ylabel('Actual Vote Share (%)', fontsize=10)
    ax2.legend()

    # 3. Residual Distribution
    ax3 = plt.subplot(2, 2, 4)
    res_data = dm['residual'] * 100
    ax3.hist(res_data, bins=45, density=True, color='mediumaquamarine', alpha=0.7)
    mu, std = res_data.mean(), res_data.std()
    xmin, xmax = ax3.get_xlim()
    x_line = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x_line, mu, std)
    ax3.plot(x_line, p, 'k', linewidth=2, label=f'Fit Normal (μ={mu:.2f}, σ={std:.2f})')
    ax3.axvline(0, color='red', lw=1, ls='--', alpha=0.5)
    ax3.set_title('Distribution of Residuals', fontsize=12)
    ax3.set_xlabel('Residual Error in Vote Share (%)', fontsize=10)
    ax3.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200)
    print(f"Saved regression dashboard → '{out_path}'")

def plot_statistical_report(results: dict, out_path: str, title: str):
    print(f"Generating statistical report image → {out_path}")
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.axis('off')
    
    text = f"=== STATISTICAL REGRESSION REPORT: {title} ===\n\n"
    text += "\n".join(results['logs']) + "\n\n"
    text += "="*60 + "\n"
    text += "=== WEIGHTED LEAST SQUARES (WLS) RESULTS ===\n"
    text += "="*60 + "\n\n"
    text += results['model_vote'].summary().as_text() + "\n"
    
    ax.text(0.01, 0.99, text, fontsize=8, va='top', ha='left', family='monospace')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    print(f"Saved statistical report → '{out_path}'")


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Korean Election Regression Model — {CFG['label']}")
    print(f"{'='*60}")

    df_census = load_census_csv(CFG['census_csv'])
    df_apt = load_apt_csv(CFG['apt_csv_glob'])
    df_dong, df_const = load_election_csv(
        CFG['result_csv'], dem_pattern=CFG['dem_pattern'], con_pattern=CFG['con_pattern']
    )

    if not df_dong.empty and not df_const.empty:
        results = run_vote_share_regression(df_dong, df_const, df_census, df_apt)
        
        plot_regression_dashboard(results, out_path=CFG['dashboard_out'], title=CFG['label'])
        report_out = CFG['dashboard_out'].replace('.png', '_report.png')
        plot_statistical_report(results, out_path=report_out, title=CFG['label'])
        
    else:
        print("[!] Election data could not be loaded. Aborting.")

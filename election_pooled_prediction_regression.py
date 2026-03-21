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
import matplotlib.pyplot as plt

try:
    import koreanize_matplotlib
except ImportError:
    print("[!] koreanize_matplotlib not found. Run: pip install koreanize-matplotlib")

# ==========================================
# CONFIGURATION
# ==========================================
ELECTION_CONFIGS = {
    21: {
        'census_csv':    '21st_election_census.csv',
        'result_csv':    '21st_election_result.csv',
        'apt_csv_glob':  '*21st_election_*_apt_price.csv',
        'dem_pattern':   r'더불어민주당',
        'con_pattern':   r'미래통합당|자유한국당',
        'label':         '21st General Election (2020)'
    },
    22: {
        'census_csv':    '22nd_election_census.csv',
        'result_csv':    '22nd_election_result.csv',
        'apt_csv_glob':  '*22nd_election_*_apt_price.csv',
        'dem_pattern':   r'더불어민주당',
        'con_pattern':   r'국민의힘',
        'label':         '22nd General Election (2024)'
    },
}

SPECIAL_DONG_NAMES = {
    '거소·선상투표', '관외사전투표', '국외부재자투표',
    '국외부재자투표(공관)', '잘못 투입·구분된 투표지',
}

GWANNAESA_LABEL = '관내사전투표'
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
# DATA LOADERS
# ==========================================
def _detect_year_prefix(df: pd.DataFrame) -> str:
    for col in df.columns:
        m = re.match(r'(\d{4}년\d{2}월)_계_총인구수', col)
        if m: return m.group(1)
    raise ValueError("Cannot detect census year prefix.")

def load_census_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path): return pd.DataFrame()
    try:
        try: df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError: df = pd.read_csv(csv_path, encoding='cp949', low_memory=False)

        prefix = _detect_year_prefix(df)
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

        ranges = [(18, 25, '1824'), (25, 30, '2529'), (30, 35, '3034'), (35, 40, '3539'),
                  (40, 45, '4044'), (45, 50, '4549'), (50, 55, '5054'), (55, 60, '5559'),
                  (60, 65, '6064'), (65, 70, '6569')]

        for g, g_str in [('남', 'm'), ('여', 'f')]:
            for r_start, r_end, r_str in ranges:
                cols = [f"{prefix}_{g}_{a}세" for a in range(r_start, r_end)]
                col_name = f'pct_{g_str}_{r_str}'
                df[col_name] = df[[c for c in cols if c in df.columns]].sum(axis=1) / df['total_voting_pop']

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
            for col in AGE_GENDER_COLS:
                row_dict[col] = row.get(col, np.nan)
            rows.append(row_dict)

        return pd.DataFrame(rows)
    except Exception as e:
        print(f"[!] Error processing census CSV {csv_path}: {e}")
        return pd.DataFrame()

def load_apt_csv(glob_pattern: str) -> pd.DataFrame:
    file_list = glob.glob(glob_pattern)
    if not file_list: return pd.DataFrame()

    df_list = []
    for file in file_list:
        try:
            try: df_temp = pd.read_csv(file, encoding='utf-8', skiprows=15)
            except UnicodeDecodeError: df_temp = pd.read_csv(file, encoding='cp949', skiprows=15)
            df_list.append(df_temp)
        except Exception: pass

    if not df_list: return pd.DataFrame()
    df = pd.concat(df_list, ignore_index=True)

    try:
        df['거래금액(만원)'] = pd.to_numeric(df['거래금액(만원)'].astype(str).str.replace(',', '').str.strip(), errors='coerce')
        df['전용면적(㎡)'] = pd.to_numeric(df['전용면적(㎡)'], errors='coerce')
        df['price_per_sqm'] = df['거래금액(만원)'] / df['전용면적(㎡)']

        def parse_loc(x):
            parts = str(x).split()
            prov = PROV_FULL_TO_SHORT.get(parts[0], parts[0]) if len(parts) > 0 else ""
            if len(parts) > 2:
                sgg = normalize_sigungu(parts[1])[0] if normalize_sigungu(parts[1]) else ""
            else:
                sgg = ""
            dong = normalize_dong_name(parts[-1]) if len(parts) > 0 else ""
            return pd.Series([prov, sgg, dong])

        df[['prov', 'sgg', 'dong_norm']] = df['시군구'].apply(parse_loc)
        apt_agg = df.groupby(['prov', 'sgg', 'dong_norm'])['price_per_sqm'].median().reset_index()
        apt_agg.rename(columns={'price_per_sqm': 'median_apt_price_sqm'}, inplace=True)
        return apt_agg
    except Exception: return pd.DataFrame()

def load_election_csv(csv_path: str, dem_pattern: str, con_pattern: str):
    try:
        try: df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError: df = pd.read_csv(csv_path, encoding='cp949', low_memory=False)
    except Exception: return pd.DataFrame()

    df['득표수']       = pd.to_numeric(df['득표수'], errors='coerce').fillna(0).astype(int)
    df['is_dem']      = df['후보자'].str.contains(dem_pattern, case=False, na=False)
    df['is_meta']     = df['후보자'].isin(META_CANDIDATES)

    dong_key  = ['시도명', '선거구명', '법정읍면동명']

    def sgg_cands_from_constituency(name):
        if not isinstance(name, str): return []
        if '_' in name: return normalize_sigungu(name.split('_', 1)[1])
        return normalize_sigungu(re.sub(r'[갑을병정무]$', '', name).strip())

    df_geo   = df[~df['법정읍면동명'].isin(SPECIAL_DONG_NAMES)].copy()
    df_votes = df_geo[~df_geo['is_meta']].copy()

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

    return df_dong

def merge_dong_with_covariates(df_election: pd.DataFrame, df_census: pd.DataFrame, df_apt: pd.DataFrame) -> pd.DataFrame:
    if not df_census.empty:
        census_lookup = {}; census_by_sgg = {}
        for _, row in df_census.iterrows():
            dnorm = row['dong_norm']
            covs = {c: row.get(c, np.nan) for c in AGE_GENDER_COLS}
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

            if covs is None: covs = {k: np.nan for k in AGE_GENDER_COLS}
            rd = row.to_dict(); rd.update(covs)
            results.append(rd)
        df_out = pd.DataFrame(results)
    else:
        df_out = df_election.copy()

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

# ==========================================
# POOLED REGRESSION ENGINE
# ==========================================
def run_pooled_shift_regression(dm_21: pd.DataFrame, dm_22: pd.DataFrame) -> dict:
    MIN_VOTES = 50

    dm_21 = dm_21[(dm_21['total_votes'] > MIN_VOTES) & (dm_21['total_dem'] > 0)].copy()
    dm_21['is_22nd'] = 0

    dm_22 = dm_22[(dm_22['total_votes'] > MIN_VOTES) & (dm_22['total_dem'] > 0)].copy()
    dm_22['is_22nd'] = 1

    df_pooled = pd.concat([dm_21, dm_22], ignore_index=True)
    df_pooled['vote_share'] = df_pooled['total_dem'] / df_pooled['total_votes']

    model_age_gender_cols = [c for c in AGE_GENDER_COLS if c != 'pct_f_4044']
    req_cols = ['vote_share', 'is_22nd', 'province_tag', 'urban_type', 'is_military', 'sum_people', 'log_apt_price'] + AGE_GENDER_COLS
    df_mod = df_pooled.dropna(subset=req_cols).copy()

    cont_cols = model_age_gender_cols + ['log_apt_price']

    # --- NEW: Print the real-world standard deviations before scaling ---
    print("\n--- Standard Deviations of Continuous Variables (1 SD = ...) ---")
    for col in cont_cols:
        raw_sd = df_mod[col].std()
        if col.startswith('pct_'):
            print(f"  {col}: {raw_sd:.4f} ({raw_sd*100:.2f}%)")
        else:
            print(f"  {col}: {raw_sd:.4f}")
    print("--------------------------------------------------------------\n")

    scaler = StandardScaler()
    df_mod[cont_cols] = scaler.fit_transform(df_mod[cont_cols])

    base_covariates = f"{' + '.join(model_age_gender_cols)} + is_military + log_apt_price + C(province_tag) + C(urban_type)"
    national_shift = "is_22nd"
    regional_shift = "is_22nd:C(province_tag)"
    cohort_shift = " + ".join([f"is_22nd:{c}" for c in model_age_gender_cols])

    formula = f"vote_share ~ {base_covariates} + {national_shift} + {regional_shift} + {cohort_shift}"

    print("Fitting Pooled DiD-Style Regression Model...")
    model = smf.wls(formula, data=df_mod, weights=df_mod['sum_people']).fit(cov_type='HC3')

    print(f"Model R²: {model.rsquared*100:.2f}%")
    df_mod['residual'] = model.resid
    df_mod['fitted']   = model.fittedvalues

    return {'df': df_mod, 'model': model}

# ==========================================
# VISUALIZATION
# ==========================================
def plot_shift_dashboard(results: dict, out_path: str):
    model = results['model']
    df_mod = results['df']

    # ----------------------------------------------------
    # 1. EXTRACT COEFFICIENTS USING JOINT VARIANCE
    # ----------------------------------------------------
    params = model.params
    cov_mat = model.cov_params()

    # Identify base covariates (exclude Intercept and pure national shift term)
    base_names = [c for c in params.index if ':' not in c and c not in ['Intercept', 'is_22nd']]

    data = []
    for b in base_names:
        # --- NEW: Multiply by 100 to convert from proportion to percentage points ---
        est_21 = params[b] * 100
        se_21 = np.sqrt(cov_mat.loc[b, b]) * 100

        # Format the interaction column name correctly for statsmodels
        shift_col = f"is_22nd:{b}"

        if shift_col in params:
            est_shift = params[shift_col] * 100
            est_22 = est_21 + est_shift
            # Joint Variance: Var(A+B) = Var(A) + Var(B) + 2Cov(A,B)
            # The standard error is calculated, then multiplied by 100 for scaling.
            var_22 = cov_mat.loc[b, b] + cov_mat.loc[shift_col, shift_col] + 2 * cov_mat.loc[b, shift_col]
            se_22 = np.sqrt(var_22) * 100
        else:
            est_22 = est_21
            se_22 = se_21

        data.append({
            'feature': b,
            'est_21': est_21, 'err_21': 1.96 * se_21,
            'est_22': est_22, 'err_22': 1.96 * se_22
        })

    df_coef = pd.DataFrame(data).set_index('feature')

    # Clean index names for presentation
    df_coef.index = df_coef.index.str.replace('C(province_tag)[T.', 'Region: ', regex=False).str.replace(']', '', regex=False)
    df_coef.index = df_coef.index.str.replace('C(urban_type)[T.', 'Urban: ', regex=False).str.replace(']', '', regex=False)
    df_coef = df_coef.sort_values('est_21')

    # ----------------------------------------------------
    # 2. PLOTTING LAYOUT
    # ----------------------------------------------------
    fig = plt.figure(figsize=(22, 24))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 1, 1])
    fig.suptitle("Pooled Electoral Regression Dashboard (21st vs 22nd Election)", fontsize=20, fontweight='bold', y=0.97)

    # SUBPLOT 1: Coefficient Dumbbell Plot (Spans both columns)
    ax_coef = fig.add_subplot(gs[0, :])
    y_pos = np.arange(len(df_coef))

    # Plot connecting line to emphasize shift direction
    for i in range(len(df_coef)):
        ax_coef.plot([df_coef['est_21'].iloc[i], df_coef['est_22'].iloc[i]], [y_pos[i]-0.15, y_pos[i]+0.15], color='gray', alpha=0.3, zorder=1)

    # Plot 21st and 22nd point estimates with Error Bars
    ax_coef.errorbar(df_coef['est_21'], y_pos - 0.15, xerr=df_coef['err_21'], fmt='o', color='dodgerblue', markersize=6, elinewidth=2, label='21st Election (2020)', zorder=2)
    ax_coef.errorbar(df_coef['est_22'], y_pos + 0.15, xerr=df_coef['err_22'], fmt='o', color='crimson', markersize=6, elinewidth=2, label='22nd Election (2024)', zorder=3)

    ax_coef.axvline(0, color='black', linestyle='--', linewidth=1)
    ax_coef.set_yticks(y_pos)
    ax_coef.set_yticklabels(df_coef.index, fontsize=11)
    ax_coef.set_title(f'Coefficient Shift Per Election (Derived from Pooled Model)', fontsize=14)
    # --- NEW: Updated label to reflect percentage points ---
    ax_coef.set_xlabel('Effect on Dem Vote Share (Percentage Points)', fontsize=12)
    ax_coef.legend(fontsize=12, loc='lower right')

    # Data separation for predictive scatter/residuals
    df_21 = df_mod[df_mod['is_22nd'] == 0]
    df_22 = df_mod[df_mod['is_22nd'] == 1]

    # SUBPLOT 2 & 3: Actual vs Predicted
    def plot_avp(ax, df_sub, title_year, color):
        x = df_sub['fitted'] * 100
        y = df_sub['vote_share'] * 100
        ax.scatter(x, y, alpha=0.3, s=15, color=color, edgecolors='none')
        mn = min(x.min(), y.min())
        mx = max(x.max(), y.max())
        ax.plot([mn, mx], [mn, mx], 'k--', lw=1.5, alpha=0.6)
        ax.set_title(f'Actual vs Predicted - {title_year}', fontsize=13)
        ax.set_xlabel('Predicted Vote Share (%)', fontsize=10)
        ax.set_ylabel('Actual Vote Share (%)', fontsize=10)

    ax_avp_21 = fig.add_subplot(gs[1, 0])
    plot_avp(ax_avp_21, df_21, '21st Election (2020)', 'dodgerblue')

    ax_avp_22 = fig.add_subplot(gs[1, 1])
    plot_avp(ax_avp_22, df_22, '22nd Election (2024)', 'crimson')

    # SUBPLOT 4 & 5: Residuals
    def plot_res(ax, df_sub, title_year, color):
        res_data = df_sub['residual'] * 100
        ax.hist(res_data, bins=45, density=True, color=color, alpha=0.6)
        mu, std = res_data.mean(), res_data.std()
        xmin, xmax = ax.get_xlim()
        x_line = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x_line, mu, std)
        ax.plot(x_line, p, 'k', linewidth=2)
        ax.axvline(0, color='red', lw=1, ls='--', alpha=0.5)
        ax.set_title(f'Residuals - {title_year} (μ={mu:.2f}, σ={std:.2f})', fontsize=13)
        ax.set_xlabel('Residual Error in Vote Share (%)', fontsize=10)

    ax_res_21 = fig.add_subplot(gs[2, 0])
    plot_res(ax_res_21, df_21, '21st Election (2020)', 'dodgerblue')

    ax_res_22 = fig.add_subplot(gs[2, 1])
    plot_res(ax_res_22, df_22, '22nd Election (2024)', 'crimson')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=200)
    print(f"Saved shift dashboard → '{out_path}'")

    # Save summary report
    report_out = out_path.replace('.png', '_report.txt')
    with open(report_out, 'w') as f:
        f.write(model.summary().as_text())
    print(f"Saved full statistical summary text → '{report_out}'")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Korean Election Pooled Shift Model (21st vs 22nd)")
    print(f"{'='*60}")

    datasets = {}
    for year, cfg in ELECTION_CONFIGS.items():
        print(f"\n--- Loading {cfg['label']} ---")
        df_cen = load_census_csv(cfg['census_csv'])
        df_apt = load_apt_csv(cfg['apt_csv_glob'])
        df_elec = load_election_csv(cfg['result_csv'], cfg['dem_pattern'], cfg['con_pattern'])

        if not df_elec.empty:
            datasets[year] = merge_dong_with_covariates(df_elec, df_cen, df_apt)
            print(f"    Merged {len(datasets[year])} districts for {year}st/nd election.")

    if 21 in datasets and 22 in datasets:
        results = run_pooled_shift_regression(datasets[21], datasets[22])
        plot_shift_dashboard(results, out_path='pooled_electoral_dashboard.png')
    else:
        print("[!] Missing data for one or both elections. Cannot run pooled model.")

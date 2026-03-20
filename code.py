import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import re
from difflib import get_close_matches
from scipy.stats import chisquare, pearsonr


# ==========================================
# KOREAN FONT SETUP
# ==========================================

def setup_korean_font():
    """
    Configure matplotlib to render Korean (Hangul) text.

    Priority order:
      1. Noto Sans/Serif CJK  – ships with most Linux distros; covers Korean
      2. NanumGothic / NanumBarunGothic – common on Korean systems
      3. Malgun Gothic – Windows Korean system font
      4. Apple SD Gothic Neo – macOS Korean system font
      5. Fallback: warn and continue (glyphs will be missing but no crash)

    Also disables the minus-sign substitution that can break axis labels.
    """
    CANDIDATES = [
        # (search substring in font name, ttc/ttf path hint)
        ('Noto Sans CJK', None),
        ('Noto Serif CJK', None),
        ('NanumGothic', None),
        ('NanumBarunGothic', None),
        ('Malgun Gothic', None),
        ('Apple SD Gothic Neo', None),
        ('UnDotum', None),
        ('Baekmuk', None),
    ]

    # Try fonts already registered in matplotlib's font manager
    all_names = {f.name for f in fm.fontManager.ttflist}
    for substr, _ in CANDIDATES:
        match = next((n for n in all_names if substr in n), None)
        if match:
            matplotlib.rcParams['font.family'] = match
            matplotlib.rcParams['axes.unicode_minus'] = False
            print(f"    Korean font set to: '{match}'")
            return

    # Try loading Noto CJK directly from common install paths
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

    # Last resort: try apt-get install fonts-nanum (works on Colab / Debian / Ubuntu)
    print("    No Korean font found – attempting: apt-get install fonts-nanum ...")
    import subprocess, shutil
    if shutil.which('apt-get'):
        try:
            subprocess.run(
                ['apt-get', 'install', '-y', '-q', 'fonts-nanum'],
                check=True, capture_output=True,
            )
            # Rebuild matplotlib font cache so newly installed fonts are visible
            fm.fontManager.__init__()
            all_names = {f.name for f in fm.fontManager.ttflist}
            match = next((n for n in all_names if 'Nanum' in n), None)
            if match:
                matplotlib.rcParams['font.family'] = match
                matplotlib.rcParams['axes.unicode_minus'] = False
                print(f"    Korean font installed and set to: '{match}'")
                return
        except Exception as e:
            print(f"    apt-get failed: {e}")

    print("    [!] Could not install a Korean font automatically.\n"
          "        In Colab run:  !apt-get install -y fonts-nanum\n"
          "        then restart the runtime and re-run.")

# ==========================================
# CONFIGURATION
# ==========================================
# Set ELECTION_NUM to 21 or 22 to switch datasets.

ELECTION_NUM = 22

ELECTION_CONFIGS = {
    21: {
        'census_csv':    '21st_election_census.csv',
        'result_csv':    '21st_election_result.csv',
        'dem_pattern':   r'더불어민주당',
        'con_pattern':   r'미래통합당|자유한국당',
        'label':         '21st General Election (2020)',
        'dashboard_out': 'mega_forensics_dashboard_21st.png',
    },
    22: {
        'census_csv':    '22nd_election_census.csv',
        'result_csv':    '22nd_election_result.csv',
        'dem_pattern':   r'더불어민주당',
        'con_pattern':   r'국민의힘',
        'label':         '22nd General Election (2024)',
        'dashboard_out': 'mega_forensics_dashboard_22nd.png',
    },
}

CFG = ELECTION_CONFIGS[ELECTION_NUM]

# ─── Row-classification constants ────────────────────────────────────────────

# 법정읍면동명 values that have no geographic dong home
SPECIAL_DONG_NAMES = {
    '거소·선상투표',
    '관외사전투표',
    '국외부재자투표',
    '국외부재자투표(공관)',
    '잘못 투입·구분된 투표지',
}

GWANNAESA_LABEL = '관내사전투표'   # in-district early vote precinct label
GWANOE_LABEL    = '관외사전투표'   # out-of-district early vote (constituency total)
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
    if not isinstance(name, str):
        return ""
    name = re.sub(r'\(.*?\)', '', name)
    name = name.strip()
    name = name.replace('.', '·')
    name = re.sub(r'제(\d)', r'\1', name)
    name = re.sub(r'·\d+', '', name)
    name = re.sub(r'(\d+)(동|읍|면)$', r'\2', name)
    name = re.sub(r'\s+', ' ', name)
    return name


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
    gu_count     = sum(1 for t in tokens if t[-1] == '구' and len(t) >= 2)
    if si_gun_count >= 2 or (si_gun_count == 0 and gu_count >= 2):
        ordered = tokens
    else:
        ordered = list(reversed(tokens))
    candidates = []
    for t in ordered:
        key = re.sub(r'[시군구]$', '', t).strip()
        if key and key not in candidates:
            candidates.append(key)
    return candidates


# ==========================================
# 1. DEMOGRAPHIC CENSUS LOADER
# ==========================================

def _detect_year_prefix(df: pd.DataFrame) -> str:
    for col in df.columns:
        m = re.match(r'(\d{4}년\d{2}월)_계_총인구수', col)
        if m:
            return m.group(1)
    raise ValueError("Cannot detect census year prefix.")


def load_census_csv(csv_path: str) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per census dong (행정구역 leaf node).
    Columns: sgg_candidates, primary_sgg, dong_norm, dong_raw,
             demographic_propensity  (share of 40-59 yr-olds among voters 18+).
    """
    print(f"\n--- [1/3] Loading Demographic Census Data ({csv_path}) ---")
    if not os.path.exists(csv_path):
        print("[!] Census CSV not found – demographic merge will be skipped.")
        return pd.DataFrame()
    try:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='cp949', low_memory=False)

        prefix = _detect_year_prefix(df)
        print(f"    Detected census year prefix: '{prefix}'")

        propensity_cols = [f"{prefix}_계_{a}세" for a in range(40, 60)]
        voting_age_cols = ([f"{prefix}_계_{a}세" for a in range(18, 100)] +
                           [f"{prefix}_계_100세 이상"])

        for col in propensity_cols + voting_age_cols:
            if col in df.columns:
                df[col] = (df[col].astype(str)
                                  .str.replace(',', '', regex=False)
                                  .pipe(pd.to_numeric, errors='coerce'))

        df['target_pop']       = df[[c for c in propensity_cols if c in df.columns]].sum(axis=1)
        df['total_voting_pop'] = df[[c for c in voting_age_cols  if c in df.columns]].sum(axis=1)
        df = df[df['total_voting_pop'] > 0].copy()
        df['demographic_propensity'] = df['target_pop'] / df['total_voting_pop']

        def extract_census_keys(admin_name):
            if not isinstance(admin_name, str):
                return [], ""
            clean = re.sub(r'\(.*?\)', '', admin_name).strip()
            parts = clean.split()
            dong_norm = normalize_dong_name(parts[-1]) if parts else ""
            sgg_cands = []
            for token in reversed(parts[:-1]):
                for c in normalize_sigungu(token):
                    if c not in sgg_cands:
                        sgg_cands.append(c)
            return sgg_cands, dong_norm

        rows = []
        for _, row in df.iterrows():
            sgg_cands, dong_norm = extract_census_keys(row['행정구역'])
            rows.append({
                'sgg_candidates':         sgg_cands,
                'primary_sgg':            sgg_cands[0] if sgg_cands else "",
                'dong_norm':              dong_norm,
                'dong_raw':               row['행정구역'],
                'demographic_propensity': row['demographic_propensity'],
            })

        census = pd.DataFrame(rows)
        print(f"    Loaded {len(census):,} census rows.")
        return census

    except Exception as e:
        print(f"[!] Error processing census CSV: {e}")
        import traceback; traceback.print_exc()
        return pd.DataFrame()


# ==========================================
# 2. ELECTION CSV LOADER
# ==========================================

def load_election_csv(csv_path: str,
                      dem_pattern: str,
                      con_pattern: str):
    """
    Parse the NEC CSV and return three DataFrames.

    df_dong  — one row per geographic dong.
               'early' = 관내사전투표 only.
               관외사전 is NOT included (no per-dong split in the CSV).
               → Use for fine-grained dong-level forensics with census controls.

    df_const — one row per constituency (선거구명).
               'sajeong' (사전) = 관내사전 + 관외사전 combined.
               'same_day' = all 본투표 across the constituency.
               → Use for full-사전투표 analysis at the cost of coarser granularity.

    df_station — one row per polling station; used for station-level tests.

    Registration note
    ─────────────────
    Every registered voter is assigned to exactly one same-day polling station
    before the election.  Summing same-day 선거인수 gives the complete dong
    voter roll.  관내/관외사전 선거인수 rows record early-voter *turnout* (not
    a separate roster) and must NOT be added to sum_people.
    """
    print(f"\n--- [2/3] Loading Election Result Data ({csv_path}) ---")
    try:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='cp949', low_memory=False)
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
        if not isinstance(name, str):
            return []
        if '_' in name:
            return normalize_sigungu(name.split('_', 1)[1])
        return normalize_sigungu(re.sub(r'[갑을병정무]$', '', name).strip())

    # ── geographic rows only (excludes 관외사전, 거소, 국외, etc.) ─────────────
    df_geo   = df[~df['법정읍면동명'].isin(SPECIAL_DONG_NAMES)].copy()
    df_votes = df_geo[~df_geo['is_meta']].copy()

    # =========================================================================
    # A.  DONG-LEVEL  (관내사전 vs 본투표)
    # =========================================================================

    gn_dem = (df_votes[df_votes['is_dem'] &  df_votes['is_gwannaesa']]
              .groupby(dong_key)['득표수'].sum().reset_index(name='gwannaesa_dem'))
    gn_tot = (df_votes[df_votes['is_gwannaesa']]
              .groupby(dong_key)['득표수'].sum().reset_index(name='gwannaesa_total'))
    sd_dem = (df_votes[df_votes['is_dem'] & ~df_votes['is_gwannaesa']]
              .groupby(dong_key)['득표수'].sum().reset_index(name='same_day_dem'))
    sd_tot = (df_votes[~df_votes['is_gwannaesa']]
              .groupby(dong_key)['득표수'].sum().reset_index(name='same_day_total'))

    # Registration: same-day station 선거인수 (complete dong voter roll)
    sum_people_dong = (
        df_geo[~df_geo['is_gwannaesa'] & (df_geo['후보자'] == '선거인수')]
        .groupby(dong_key)['득표수'].sum().reset_index(name='sum_people')
    )
    # Geographic turnout: 관내사전 + 본투표 (관외사전 excluded here)
    sum_vote_geo = (
        df_geo[df_geo['후보자'] == '투표수']
        .groupby(dong_key)['득표수'].sum().reset_index(name='sum_vote_geo')
    )

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

    print(f"    Dong-level rows  (관내사전 vs 본투표)    : {len(df_dong):>5,}")

    # =========================================================================
    # B.  CONSTITUENCY-LEVEL  (사전투표 전체 vs 본투표)
    # =========================================================================
    # 사전 = 관내사전 (per-dong, aggregated to constituency) +
    #         관외사전 (already constituency-level in CSV)

    # -- 관내사전 → constituency --
    gn_dem_c  = (df_votes[df_votes['is_dem'] &  df_votes['is_gwannaesa']]
                 .groupby(const_key)['득표수'].sum().reset_index(name='gwannaesa_dem'))
    gn_tot_c  = (df_votes[df_votes['is_gwannaesa']]
                 .groupby(const_key)['득표수'].sum().reset_index(name='gwannaesa_total'))
    gn_turn_c = (df_geo[df_geo['is_gwannaesa'] & (df_geo['후보자'] == '투표수')]
                 .groupby(const_key)['득표수'].sum().reset_index(name='gwannaesa_turnout'))

    # -- 관외사전 (already at constituency level) --
    df_gw  = df[df['법정읍면동명'] == GWANOE_LABEL]
    df_gw_v= df_gw[~df_gw['is_meta']]
    go_dem_c  = (df_gw_v[df_gw_v['is_dem']]
                 .groupby(const_key)['득표수'].sum().reset_index(name='gwanoe_dem'))
    go_tot_c  = (df_gw_v
                 .groupby(const_key)['득표수'].sum().reset_index(name='gwanoe_total'))
    go_turn_c = (df_gw[df_gw['후보자'] == '투표수']
                 .groupby(const_key)['득표수'].sum().reset_index(name='gwanoe_turnout'))

    # -- 본투표 → constituency --
    sd_dem_c  = (df_votes[df_votes['is_dem'] & ~df_votes['is_gwannaesa']]
                 .groupby(const_key)['득표수'].sum().reset_index(name='same_day_dem'))
    sd_tot_c  = (df_votes[~df_votes['is_gwannaesa']]
                 .groupby(const_key)['득표수'].sum().reset_index(name='same_day_total'))
    sd_turn_c = (df_geo[~df_geo['is_gwannaesa'] & (df_geo['후보자'] == '투표수')]
                 .groupby(const_key)['득표수'].sum().reset_index(name='same_day_turnout'))

    # -- Registration per constituency (from dong-level) --
    reg_c = (df_dong.groupby(const_key)['sum_people']
             .sum().reset_index(name='sum_people'))

    df_const = reg_c.copy()
    for frame in (gn_dem_c, gn_tot_c, gn_turn_c,
                  go_dem_c, go_tot_c, go_turn_c,
                  sd_dem_c, sd_tot_c, sd_turn_c):
        df_const = df_const.merge(frame, on=const_key, how='left')
    df_const = df_const.fillna(0)

    # Combined 사전 totals
    df_const['sajeong_dem']      = df_const['gwannaesa_dem']     + df_const['gwanoe_dem']
    df_const['sajeong_total']    = df_const['gwannaesa_total']   + df_const['gwanoe_total']
    df_const['sajeong_turnout']  = df_const['gwannaesa_turnout'] + df_const['gwanoe_turnout']
    df_const['total_turnout']    = df_const['sajeong_turnout']   + df_const['same_day_turnout']

    df_const['sgg_candidates'] = df_const['선거구명'].apply(sgg_cands_from_constituency)
    df_const['primary_sgg']    = df_const['sgg_candidates'].apply(lambda x: x[0] if x else "")
    df_const['province_tag']   = df_const['시도명'].map(PROV_FULL_TO_SHORT).fillna(df_const['시도명'])

    print(f"    Constituency rows (사전 전체 vs 본투표) : {len(df_const):>5,}")

    # =========================================================================
    # C.  STATION-LEVEL
    # =========================================================================
    sk = ['시도명', '선거구명', '법정읍면동명', '투표구명']

    st_r = (df[df['후보자'] == '선거인수'].groupby(sk)['득표수'].sum().reset_index(name='sum_people'))
    st_v = (df[df['후보자'] == '투표수'  ].groupby(sk)['득표수'].sum().reset_index(name='sum_vote'))
    st_i = (df[df['후보자'] == '무효 투표수'].groupby(sk)['득표수'].sum().reset_index(name='sum_invalid'))
    st_d = (df[df['is_dem']].groupby(sk)['득표수'].sum().reset_index(name='dem_votes'))
    st_c = (df[df['is_con']].groupby(sk)['득표수'].sum().reset_index(name='con_votes'))

    df_station = st_r.copy()
    for frame in (st_v, st_i, st_d, st_c):
        df_station = df_station.merge(frame, on=sk, how='outer')
    df_station = df_station.fillna(0)

    df_station['is_early']    = df_station['투표구명'] == GWANNAESA_LABEL
    df_station['region']      = df_station['시도명'] + ' ' + df_station['선거구명']
    df_station['invalid_rate']= (df_station['sum_invalid'] /
                                  df_station['sum_vote'].replace(0, np.nan))
    print(f"    Station-level rows                      : {len(df_station):>5,}")

    return df_dong, df_const, df_station


# ==========================================
# 3. MULTI-PASS DONG↔CENSUS MATCHER
# ==========================================

def merge_dong_with_census(df_election: pd.DataFrame,
                           df_census:   pd.DataFrame) -> pd.DataFrame:
    """
    7-pass name matching to attach demographic_propensity to each dong.
    Unmatched rows are dropped.
    """
    if df_census.empty:
        print("[!] Census is empty – skipping demographic merge.")
        return df_election.copy()

    census_lookup = {}; census_by_sgg = {}
    prov_dong_sum = {}; prov_dong_cnt = {}
    prov_sgg_dong_sum = {}; prov_sgg_dong_cnt = {}
    dong_frequency = {}; dong_unique_map = {}

    PROV_FULL = {
        '서울특별시': '서울', '부산광역시': '부산', '대구광역시': '대구',
        '인천광역시': '인천', '광주광역시': '광주', '대전광역시': '대전',
        '울산광역시': '울산', '세종특별자치시': '세종',
        '경기도': '경기', '강원도': '강원', '강원특별자치도': '강원',
        '충청북도': '충북', '충청남도': '충남',
        '전라북도': '전북', '전북특별자치도': '전북', '전라남도': '전남',
        '경상북도': '경북', '경상남도': '경남', '제주특별자치도': '제주',
    }

    for _, row in df_census.iterrows():
        prop = row['demographic_propensity']; dnorm = row['dong_norm']
        for sgg in row['sgg_candidates']:
            census_lookup[(sgg, dnorm)] = prop
            census_by_sgg.setdefault(sgg, []).append(dnorm)
        raw = row.get('dong_raw', '')
        first = raw.split()[0] if isinstance(raw, str) and raw.split() else ''
        prov = PROV_FULL.get(first, first)
        pk = (prov, dnorm)
        prov_dong_sum[pk] = prov_dong_sum.get(pk, 0.0) + prop
        prov_dong_cnt[pk] = prov_dong_cnt.get(pk, 0) + 1
        for sgg in row['sgg_candidates']:
            tk = (prov, sgg, dnorm)
            prov_sgg_dong_sum[tk] = prov_sgg_dong_sum.get(tk, 0.0) + prop
            prov_sgg_dong_cnt[tk] = prov_sgg_dong_cnt.get(tk, 0) + 1
        dong_frequency[dnorm]  = dong_frequency.get(dnorm, 0) + 1
        dong_unique_map[dnorm] = prop

    prov_dong_map     = {k: v / prov_dong_cnt[k]     for k, v in prov_dong_sum.items()}
    prov_sgg_dong_map = {k: v / prov_sgg_dong_cnt[k] for k, v in prov_sgg_dong_sum.items()}
    dong_unique_map   = {d: p for d, p in dong_unique_map.items() if dong_frequency[d] == 1}

    results = []; stats = {f'pass{i}': 0 for i in range(1, 8)}; stats['unmatched'] = 0
    fallback_log = []

    for _, row in df_election.iterrows():
        propensity = None; match_pass = None
        dk = row['dong_norm']
        sgc = row['sgg_candidates'] if isinstance(row['sgg_candidates'], list) else [row['primary_sgg']]

        if (row['primary_sgg'], dk) in census_lookup:
            propensity, match_pass = census_lookup[(row['primary_sgg'], dk)], 'pass1'
        if propensity is None and '·' in dk:
            k1b = (row['primary_sgg'], dk.replace('·', ''))
            if k1b in census_lookup:
                propensity, match_pass = census_lookup[k1b], 'pass1'
        if propensity is None:
            for sgg in sgc[1:]:
                if (sgg, dk) in census_lookup:
                    propensity, match_pass = census_lookup[(sgg, dk)], 'pass2'; break
        if propensity is None:
            for sgg in sgc:
                pool = census_by_sgg.get(sgg, [])
                if pool:
                    m = get_close_matches(dk, pool, n=1, cutoff=0.82)
                    if m and (sgg, m[0]) in census_lookup:
                        propensity, match_pass = census_lookup[(sgg, m[0])], 'pass3'; break
        if propensity is None and dk in dong_unique_map:
            propensity, match_pass = dong_unique_map[dk], 'pass4'
        if propensity is None:
            prov = row.get('province_tag', '')
            if prov and (prov, dk) in prov_dong_map:
                propensity, match_pass = prov_dong_map[(prov, dk)], 'pass5'
        if propensity is None:
            prov = row.get('province_tag', '')
            if prov:
                for sgg in sgc:
                    if (prov, sgg, dk) in prov_sgg_dong_map:
                        propensity, match_pass = prov_sgg_dong_map[(prov, sgg, dk)], 'pass6'; break
        if propensity is None and '·' in dk:
            hits = []
            for comp in dk.split('·'):
                if not re.search(r'(동|읍|면|가)$', comp):
                    comp += '동'
                cn = normalize_dong_name(comp)
                prov = row.get('province_tag', '')
                for sgg in sgc:
                    if (sgg, cn) in census_lookup:
                        hits.append(census_lookup[(sgg, cn)]); break
                    if prov and (prov, sgg, cn) in prov_sgg_dong_map:
                        hits.append(prov_sgg_dong_map[(prov, sgg, cn)]); break
            if hits:
                propensity, match_pass = sum(hits) / len(hits), 'pass7'

        if propensity is None:
            stats['unmatched'] += 1
        else:
            stats[match_pass] += 1
            if match_pass != 'pass1':
                fallback_log.append({
                    'pass': match_pass, 'area2_name': row.get('area2_name', ''),
                    'area3_name': row.get('name', dk), 'dong_norm': dk,
                    'sgg_cands': sgc, 'province': row.get('province_tag', ''),
                })
        rd = row.to_dict()
        rd['demographic_propensity'] = propensity
        rd['match_pass']             = match_pass
        results.append(rd)

    df_out = pd.DataFrame(results)
    df_out = df_out[df_out['demographic_propensity'].notna()].copy()

    total = len(df_election)
    print(f"\n  ┌─ Dong Census-Matching ─────────────────────")
    for i in range(1, 8):
        print(f"  │  Pass {i}: {stats[f'pass{i}']:>6,}")
    print(f"  │  Unmatched : {stats['unmatched']:>6,}  ({stats['unmatched']/total*100:.1f}%)")
    if stats['unmatched'] > 0:
        all_census_dongs = {k[1] for k in census_lookup}
        for r in [r for r in results if r['match_pass'] is None][:20]:
            flag = "sgg-mismatch" if r['dong_norm'] in all_census_dongs else "NOT-IN-CENSUS"
            print(f"    [{flag}]  sgg={r['sgg_candidates']}  dong='{r['dong_norm']}'")
    print(f"  └────────────────────────────────────────────")
    return df_out


# ==========================================
# 4. CONSTITUENCY CENSUS MERGE
# ==========================================

def merge_const_with_census(df_const: pd.DataFrame,
                             df_dong_merged: pd.DataFrame) -> pd.DataFrame:
    """
    Constituency propensity = registration-weighted mean of dong propensities.
    Derived from df_dong_merged (already census-merged).
    """
    if df_dong_merged.empty or 'demographic_propensity' not in df_dong_merged.columns:
        print("[!] No dong propensities – skipping constituency census merge.")
        df_const = df_const.copy()
        df_const['demographic_propensity'] = np.nan
        return df_const

    const_key = ['시도명', '선거구명']
    dm = df_dong_merged.dropna(subset=['demographic_propensity']).copy()
    dm['_pw'] = dm['demographic_propensity'] * dm['sum_people']
    agg = (dm.groupby(const_key)
             .agg(_pw_sum=('_pw', 'sum'), _reg=('sum_people', 'sum'))
             .reset_index())
    agg['demographic_propensity'] = agg['_pw_sum'] / agg['_reg'].replace(0, np.nan)

    df_out   = df_const.merge(agg[const_key + ['demographic_propensity']],
                              on=const_key, how='left')
    matched  = df_out['demographic_propensity'].notna().sum()
    print(f"  Constituency census merge: {matched:,}/{len(df_out):,} matched.")
    return df_out


# ==========================================
# 5. FORENSICS ENGINE
# ==========================================

def categorize_metro(region: str) -> str:
    if not isinstance(region, str): return "Other"
    if '인천' in region or '계양' in region or '연수' in region: return "Incheon"
    if any(c in region for c in ('수원', '고양', '성남', '용인')): return "Gyeonggi"
    if '서울' in region: return "Seoul"
    return "Other"


def run_forensics(df_dong_raw:  pd.DataFrame,
                  df_const_raw: pd.DataFrame,
                  df_station:   pd.DataFrame,
                  df_census:    pd.DataFrame) -> dict:
    """
    Two-level forensic analysis.

    Level A  –  Dong  │  관내사전 vs 본투표
        Fine-grained; demographic controls via census.
        관외사전 excluded (no per-dong breakdown available).

    Level B  –  Constituency  │  사전 전체 vs 본투표
        관외사전 included; demographics aggregated from dongs.
        Also exposes the 관내사전 vs 관외사전 split within early votes.
    """
    print(f"\n--- [3/3] Forensics Suite ---")
    MIN_VOTES = 50

    # ── Level A: dong ─────────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  LEVEL A │ Dong-level  (관내사전 vs 본투표)")
    print("═"*55)

    dm = merge_dong_with_census(df_dong_raw, df_census)
    # Volume filter: enough votes in both early and same-day to be meaningful
    dm = dm[(dm['gwannaesa_total'] > MIN_VOTES) &
            (dm['same_day_total']  > MIN_VOTES)].copy()
    # Candidate filter: drop dongs where the target party fielded no candidate.
    # These produce gwannaesa_dem = same_day_dem = 0 – a mechanically zero gap
    # that is uninformative and distorts the gap distribution.
    no_cand = (dm['gwannaesa_dem'] == 0) & (dm['same_day_dem'] == 0)
    if no_cand.sum() > 0:
        affected = dm.loc[no_cand, '선거구명'].unique().tolist()
        shown = ', '.join(affected[:5]) + ('...' if len(affected) > 5 else '')
        print(f"  Dropping {no_cand.sum():,} no-candidate dongs "
              f"({len(affected)} constituencies: {shown})")
    dm = dm[~no_cand].copy()

    dm['early_pct']   = dm['gwannaesa_dem']  / dm['gwannaesa_total']
    dm['sameday_pct'] = dm['same_day_dem']   / dm['same_day_total']
    dm['gap']         = dm['early_pct'] - dm['sameday_pct']
    dm['w_gap']       = dm['gap'] / dm['demographic_propensity']
    dm['vote_share']  = ((dm['gwannaesa_dem'] + dm['same_day_dem']) /
                         (dm['gwannaesa_total'] + dm['same_day_total']))
    dm['turnout']     = dm['sum_vote_geo'] / dm['sum_people'].replace(0, np.nan)

    print(f"\n  Rows after filters: {len(dm):,}")

    v2bl    = dm['gwannaesa_dem'].astype(int).astype(str)
    v2bl    = v2bl[v2bl.str.len() >= 2]

    obs_2bl = v2bl.str[1].astype(int).value_counts().reindex(range(10), fill_value=0).sort_index()
    exp_2bl = [sum(math.log10(1 + 1/(10*k+d)) for k in range(1,10))*len(v2bl) for d in range(10)]
    chi_2bl, p_2bl = chisquare(obs_2bl, exp_2bl)
    print(f"[A1] 2BL (관내사전 dem)        χ²={chi_2bl:7.2f}  p={p_2bl:.4f}")

    obs_ld  = v2bl.str[-1].astype(int).value_counts().reindex(range(10), fill_value=0).sort_index()
    exp_ld  = [len(v2bl) / 10] * 10
    chi_ld, p_ld = chisquare(obs_ld, exp_ld)
    print(f"[A2] Last digit (관내사전 dem) χ²={chi_ld:7.2f}  p={p_ld:.4f}")

    variances = dm.groupby('primary_sgg')['vote_share'].std().dropna() * 100
    print(f"[A3] Cross-neighbourhood σ     {variances.mean():.2f}%")

    print(f"[A4] Gap 관내사전−본투표        mean={dm['gap'].mean()*100:+.2f}%  "
          f"σ={dm['gap'].std()*100:.2f}%")

    corr_a, _ = pearsonr(dm['sameday_pct'], dm['early_pct'])
    print(f"[A5] Algorithmic ratio         R²={corr_a**2:.4f}")

    # ── Level B: constituency ─────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  LEVEL B │ Constituency-level  (사전 전체 vs 본투표)")
    print("═"*55)

    cm = merge_const_with_census(df_const_raw, dm)
    cm = cm[(cm['sajeong_total']  > MIN_VOTES) &
            (cm['same_day_total'] > MIN_VOTES)].copy()
    # Candidate filter: drop constituencies where the target party had no candidate.
    no_cand_c = (cm['sajeong_dem'] == 0) & (cm['same_day_dem'] == 0)
    if no_cand_c.sum() > 0:
        shown = ', '.join(cm.loc[no_cand_c, '선거구명'].tolist()[:5])
        print(f"  Dropping {no_cand_c.sum():,} no-candidate constituencies: {shown}")
    cm = cm[~no_cand_c].copy()

    cm['early_pct']      = cm['sajeong_dem']     / cm['sajeong_total']
    cm['sameday_pct']    = cm['same_day_dem']    / cm['same_day_total']
    cm['gap']            = cm['early_pct'] - cm['sameday_pct']
    cm['w_gap']          = cm['gap'] / cm['demographic_propensity']
    cm['gwannaesa_pct']  = cm['gwannaesa_dem']   / cm['gwannaesa_total'].replace(0, np.nan)
    cm['gwanoe_pct']     = cm['gwanoe_dem']      / cm['gwanoe_total'].replace(0, np.nan)
    cm['gwannaesa_gap']  = cm['gwannaesa_pct']   - cm['sameday_pct']
    cm['gwanoe_gap']     = cm['gwanoe_pct']      - cm['sameday_pct']
    cm['gap_shift']      = cm['gap'] - cm['gwannaesa_gap']  # effect of including 관외사전
    cm['vote_share']     = ((cm['sajeong_dem'] + cm['same_day_dem']) /
                            (cm['sajeong_total'] + cm['same_day_total']))
    cm['turnout']        = cm['total_turnout'] / cm['sum_people'].replace(0, np.nan)

    print(f"\n  Rows after noise filter: {len(cm):,}")

    vb2bl    = cm['sajeong_dem'].astype(int).astype(str)
    vb2bl    = vb2bl[vb2bl.str.len() >= 2]
    obs_2bl_b= vb2bl.str[1].astype(int).value_counts().reindex(range(10), fill_value=0).sort_index()
    exp_2bl_b= [sum(math.log10(1+1/(10*k+d)) for k in range(1,10))*len(vb2bl) for d in range(10)]
    chi_2bl_b, p_2bl_b = chisquare(obs_2bl_b, exp_2bl_b)
    print(f"[B1] 2BL (사전 dem)            χ²={chi_2bl_b:7.2f}  p={p_2bl_b:.4f}")

    print(f"[B2] Gap 사전−본투표            mean={cm['gap'].mean()*100:+.2f}%  "
          f"σ={cm['gap'].std()*100:.2f}%")

    corr_b, _ = pearsonr(cm['sameday_pct'], cm['early_pct'])
    print(f"[B3] Algorithmic ratio         R²={corr_b**2:.4f}")

    # B4: Gap decomposition — 관내사전 vs 관외사전 gap (vs 본투표)
    valid_go = cm.dropna(subset=['gwanoe_pct'])
    print(f"[B4] Gap decomposition (vs 본투표):")
    print(f"     관내사전  mean={cm['gwannaesa_gap'].mean()*100:+.2f}%  "
          f"σ={cm['gwannaesa_gap'].std()*100:.2f}%")
    print(f"     관외사전  mean={valid_go['gwanoe_gap'].mean()*100:+.2f}%  "
          f"σ={valid_go['gwanoe_gap'].std()*100:.2f}%  (n={len(valid_go):,})")
    print(f"     gap shift from including 관외사전: "
          f"mean={cm['gap_shift'].mean()*100:+.2f}%  "
          f"σ={cm['gap_shift'].std()*100:.2f}%")

    # B5: Metro aggregate (63:36 theory) — 관내사전 stations
    df_station['metro_zone'] = df_station['region'].apply(categorize_metro)
    early_st  = df_station[df_station['is_early']].copy()
    metro_agg = early_st.groupby('metro_zone')[['dem_votes', 'con_votes']].sum()
    metro_agg['total']    = metro_agg['dem_votes'] + metro_agg['con_votes']
    metro_agg['dem_pct']  = metro_agg['dem_votes'] / metro_agg['total'] * 100
    print(f"[B5] Metro aggregates (관내사전 stations):")
    for zone, row in metro_agg.iterrows():
        if zone != "Other" and row['total'] > 0:
            print(f"     {zone:10s}  Dem {row['dem_pct']:.2f}% | "
                  f"Con {100-row['dem_pct']:.2f}%")
    micro_std = (early_st['dem_votes'] /
                 (early_st['dem_votes'] + early_st['con_votes']).replace(0, np.nan)
                 ).std() * 100
    print(f"     Station-level σ: {micro_std:.2f}%")

    # B6: Invalid vote correlation
    ec = early_st.dropna(subset=['invalid_rate']).copy()
    ec['micro_dem'] = (ec['dem_votes'] /
                       (ec['dem_votes'] + ec['con_votes']).replace(0, np.nan)) * 100
    ec = ec.dropna(subset=['micro_dem'])
    if len(ec) > 10:
        inv_r, inv_p = pearsonr(ec['invalid_rate'], ec['micro_dem'])
        print(f"[B6] Invalid-vote correlation  r={inv_r:.4f}  p={inv_p:.4f}")

    return {
        'dong':  {
            'df': dm, 'obs_ld': obs_ld, 'exp_ld': exp_ld,
            'variances': variances, 'r2': corr_a**2,
        },
        'const': {
            'df': cm, 'r2': corr_b**2,
        },
    }


# ==========================================
# 6. DASHBOARD  (3 × 4 grid)
# ==========================================

def plot_dashboard(results: dict, out_path: str, title: str):
    print(f"\nGenerating dashboard → {out_path}")
    dm       = results['dong']['df']
    cm       = results['const']['df']
    obs_ld   = results['dong']['obs_ld']
    exp_ld   = results['dong']['exp_ld']
    variances= results['dong']['variances']
    r2_dong  = results['dong']['r2']
    r2_const = results['const']['r2']

    fig, axes = plt.subplots(3, 4, figsize=(22, 15))
    fig.suptitle(f"Election Forensics Dashboard  –  {title}",
                 fontsize=15, fontweight='bold', y=0.995)

    LEVEL_A = '관내사전 vs 본투표 (dong)'
    LEVEL_B = '사전 전체 vs 본투표 (constituency)'

    def hist_gap(ax, data, label, color):
        data = data.dropna()
        ax.hist(data * 100, bins=40, color=color, alpha=0.65, edgecolor='none')
        ax.axvline(data.mean() * 100, color='black', lw=1.8, ls='--',
                   label=f'μ = {data.mean()*100:+.2f}%\nσ = {data.std()*100:.2f}%')
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Gap (%)', fontsize=8)
        ax.legend(fontsize=7, handlelength=1)

    def scatter_ratio(ax, x, y, r2, label):
        mn = min(x.min(), y.min()) * 100 - 1
        mx = max(x.max(), y.max()) * 100 + 1
        ax.scatter(x * 100, y * 100, alpha=0.25, s=12,
                   color='royalblue', edgecolors='none')
        ax.plot([mn, mx], [mn, mx], 'r--', lw=1.2, label='1:1')
        ax.set_title(f'{label}  R²={r2:.4f}', fontsize=9)
        ax.set_xlabel('본투표 dem share (%)', fontsize=8)
        ax.set_ylabel('사전 dem share (%)', fontsize=8)
        ax.legend(fontsize=7)

    # ── Row 0: raw gap + demographic-weighted gap ─────────────────────────────
    hist_gap(axes[0, 0], dm['gap'],      f'Raw gap\n{LEVEL_A}',       'tomato')
    hist_gap(axes[0, 1], cm['gap'],      f'Raw gap\n{LEVEL_B}',       'darkorange')
    hist_gap(axes[0, 2], dm['w_gap'],    f'Weighted gap\n{LEVEL_A}',  'seagreen')
    hist_gap(axes[0, 3], cm['w_gap'],    f'Weighted gap\n{LEVEL_B}',  'teal')

    # ── Row 1: algorithmic ratio + election fingerprint ───────────────────────
    scatter_ratio(axes[1, 0], dm['sameday_pct'], dm['early_pct'], r2_dong,
                  f'Algorithmic ratio\n{LEVEL_A}')
    scatter_ratio(axes[1, 1], cm['sameday_pct'], cm['early_pct'], r2_const,
                  f'Algorithmic ratio\n{LEVEL_B}')

    fp = dm[(dm['turnout'] > 0) & (dm['turnout'] <= 1.0)]
    hb = axes[1, 2].hexbin(fp['turnout']*100, fp['vote_share']*100,
                            gridsize=28, cmap='inferno', mincnt=1)
    fig.colorbar(hb, ax=axes[1, 2], label='dongs')
    axes[1, 2].set_title(f'Election fingerprint\n{LEVEL_A}', fontsize=9)
    axes[1, 2].set_xlabel('Turnout (%)', fontsize=8)
    axes[1, 2].set_ylabel('Dem vote share (%)', fontsize=8)

    fpc = cm[(cm['turnout'] > 0) & (cm['turnout'] <= 1.0)]
    hb2 = axes[1, 3].hexbin(fpc['turnout']*100, fpc['vote_share']*100,
                             gridsize=18, cmap='inferno', mincnt=1)
    fig.colorbar(hb2, ax=axes[1, 3], label='const.')
    axes[1, 3].set_title(f'Election fingerprint\n{LEVEL_B}', fontsize=9)
    axes[1, 3].set_xlabel('Turnout (%)', fontsize=8)
    axes[1, 3].set_ylabel('Dem vote share (%)', fontsize=8)

    # ── Row 2: digit tests + gap decomposition ────────────────────────────────
    axes[2, 0].bar(range(10), obs_ld, color='salmon', alpha=0.8, edgecolor='white', label='Observed')
    axes[2, 0].plot(range(10), exp_ld, 'k--', lw=1.5, label='Expected uniform')
    axes[2, 0].set_title(f'Last-digit test\n관내사전 dem votes (dong)', fontsize=9)
    axes[2, 0].set_xticks(range(10)); axes[2, 0].legend(fontsize=7)

    axes[2, 1].hist(variances, bins=25, color='mediumpurple', alpha=0.75, edgecolor='none')
    axes[2, 1].set_title(f'Vote-share variance by city\n{LEVEL_A}', fontsize=9)
    axes[2, 1].set_xlabel('σ (%)', fontsize=8)

    # 관내사전 gap vs 관외사전 gap scatter (constituency)
    vgo = cm.dropna(subset=['gwanoe_pct'])
    axes[2, 2].scatter(vgo['gwannaesa_gap']*100, vgo['gwanoe_gap']*100,
                       alpha=0.35, s=14, color='steelblue', edgecolors='none')
    lims = [min(vgo['gwannaesa_gap'].min(), vgo['gwanoe_gap'].min())*100 - 1,
            max(vgo['gwannaesa_gap'].max(), vgo['gwanoe_gap'].max())*100 + 1]
    axes[2, 2].plot(lims, lims, 'r--', lw=1.2, label='1:1')
    axes[2, 2].set_title('관내사전 gap vs 관외사전 gap\n(both vs 본투표, constituency)', fontsize=9)
    axes[2, 2].set_xlabel('관내사전 − 본투표 (%)', fontsize=8)
    axes[2, 2].set_ylabel('관외사전 − 본투표 (%)', fontsize=8)
    axes[2, 2].legend(fontsize=7)

    # Gap shift: adding 관외사전
    hist_gap(axes[2, 3], cm['gap_shift'],
             'Gap shift from adding 관외사전\n(constituency)', 'goldenrod')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=200)
    print(f"Saved '{out_path}'")


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  Korean Election Forensics — {CFG['label']}")
    print(f"{'='*60}")

    setup_korean_font()

    df_census                      = load_census_csv(CFG['census_csv'])
    df_dong, df_const, df_station  = load_election_csv(
        CFG['result_csv'],
        dem_pattern=CFG['dem_pattern'],
        con_pattern=CFG['con_pattern'],
    )

    if not df_dong.empty and not df_const.empty:
        results = run_forensics(df_dong, df_const, df_station, df_census)
        plot_dashboard(results, out_path=CFG['dashboard_out'], title=CFG['label'])
    else:
        print("[!] Election data could not be loaded. Aborting.")

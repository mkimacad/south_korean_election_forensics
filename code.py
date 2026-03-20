import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sqlite3
import os
import re
from difflib import get_close_matches
from scipy.stats import chisquare, pearsonr

# ==========================================
# SHARED NAME NORMALIZATION UTILITIES
# ==========================================

def normalize_dong_name(name: str) -> str:
    """
    Canonical normalization applied identically to BOTH census and election names
    before any matching attempt.

    Steps (order matters):
      1. Remove parenthetical annotations  e.g. "역삼동(일부)" -> "역삼동"
      2. Strip leading/trailing whitespace
      3. Remove the numeric sub-district suffix that appears in census but not
         always in election data, or vice-versa  e.g. "역삼1동" -> "역삼동"
         Rule: if the name ends with <digit(s)><동|읍|면>, drop the digits.
      4. Collapse internal whitespace to a single space
    """
    if not isinstance(name, str):
        return ""
    name = re.sub(r'\(.*?\)', '', name)              # step 1: drop code suffix
    name = name.strip()                               # step 2
    name = name.replace('.', '·')                     # step 3: census uses '.' election uses '·'
    name = re.sub(r'·\d+', '', name)                 # step 4: "상계3·4동"->"상계3동", "종로1·2·3·4가동"->"종로1가동"
    name = re.sub(r'제?(\d+)(동|읍|면)$', r'\2', name) # step 5: "창제1동"->"창동", "역삼1동"->"역삼동"
    name = re.sub(r'\s+', ' ', name)                 # step 6
    return name


def normalize_sigungu(name: str) -> str:
    """
    Strip trailing 시/군/구 AND common province prefixes so that
    '종로구' and '종로' both map to '종로',
    and compound names like '수원시장안구' are split intelligently.

    Returns a *list* of candidate keys (most specific first) so the
    caller can try each one.
    """
    if not isinstance(name, str):
        return []
    name = name.strip()
    # Remove parentheticals
    name = re.sub(r'\(.*?\)', '', name).strip()
    candidates = []

    # If the string contains multiple 시/군/구 segments (e.g. "수원시장안구"),
    # split them out so we can try both "장안구"→"장안" and "수원시"→"수원".
    parts = re.split(r'(?<=[시군구])', name)
    parts = [p.strip() for p in parts if p.strip()]

    for part in reversed(parts):           # most-specific (smallest) unit first
        stripped = re.sub(r'[시군구]$', '', part).strip()
        if stripped:
            candidates.append(stripped)

    # Also keep the raw last-word fallback
    raw_last = re.sub(r'[시군구갑을병정무]$', '', name.split()[-1]).strip()
    if raw_last and raw_last not in candidates:
        candidates.append(raw_last)

    return candidates


# ==========================================
# 1. DEMOGRAPHIC CENSUS EXTRACTOR
# ==========================================

def fetch_kosis_demographics_csv(csv_path="data.csv"):
    print(f"\n--- [1/3] Loading Demographic Census Data ({csv_path}) ---")
    if not os.path.exists(csv_path):
        print("[!] Census CSV not found – demographic merge will be skipped.")
        return pd.DataFrame()

    try:
        try:
            df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding='cp949', low_memory=False)

        propensity_cols   = [f"2020년04월_계_{age}세" for age in range(40, 60)]
        voting_age_cols   = [f"2020년04월_계_{age}세" for age in range(18, 100)] + ['2020년04월_계_100세 이상']

        for col in propensity_cols + voting_age_cols:
            if col in df.columns:
                df[col] = (df[col].astype(str)
                                  .str.replace(',', '', regex=False)
                                  .astype(float))

        df['target_pop']        = df[propensity_cols].sum(axis=1)
        df['total_voting_pop']  = df[voting_age_cols].sum(axis=1)
        df = df[df['total_voting_pop'] > 0].copy()
        df['demographic_propensity'] = df['target_pop'] / df['total_voting_pop']

        def extract_census_keys(admin_name):
            """
            Parse a census 행정구역 string such as
            '서울특별시 종로구 청운효자동'
            into normalised (si_gun_gu_key, dong_key) pairs.
            """
            if not isinstance(admin_name, str):
                return [], ""
            clean = re.sub(r'\(.*?\)', '', admin_name).strip()
            parts = clean.split()

            # The dong is always the last token
            dong_raw  = parts[-1] if parts else ""
            dong_norm = normalize_dong_name(dong_raw)

            # The si/gun/gu candidates come from every preceding token
            sgg_candidates = []
            for token in reversed(parts[:-1]):
                for cand in normalize_sigungu(token):
                    if cand not in sgg_candidates:
                        sgg_candidates.append(cand)

            return sgg_candidates, dong_norm

        rows = []
        for _, row in df.iterrows():
            sgg_cands, dong_norm = extract_census_keys(row['행정구역'])
            rows.append({
                'sgg_candidates':          sgg_cands,
                'primary_sgg':             sgg_cands[0] if sgg_cands else "",
                'dong_norm':               dong_norm,
                'dong_raw':                row['행정구역'],
                'demographic_propensity':  row['demographic_propensity'],
            })

        census = pd.DataFrame(rows)
        print(f"    Loaded {len(census):,} census rows.")
        return census

    except Exception as e:
        print(f"[!] Error processing CSV: {e}")
        return pd.DataFrame()


# ==========================================
# 2. ELECTION DATABASE EXTRACTORS
# ==========================================

def fetch_election_data():
    """Extracts BOTH Dong-level (area3) and Station-level (area4) data."""
    db_file = "korea_election_regional_21_kor.sqlite"
    print(f"--- [2/3] Extracting Election Data from SQLite ---")

    conn = sqlite3.connect(db_file)
    area2     = pd.read_sql_query("SELECT * FROM area2", conn)
    area3     = pd.read_sql_query("SELECT * FROM area3", conn)
    area4     = pd.read_sql_query("SELECT * FROM area4", conn)
    area1     = pd.read_sql_query("SELECT * FROM area1", conn)
    party     = pd.read_sql_query("SELECT * FROM party", conn)
    candidate = pd.read_sql_query("SELECT * FROM candidate", conn)
    vote      = pd.read_sql_query("SELECT * FROM vote", conn)

    # Map candidates → parties
    cand_to_party = dict(zip(candidate['uid'], candidate['party']))
    vote['party']  = vote['candidate'].map(cand_to_party).map(dict(zip(party['uid'], party['name'])))
    vote['is_dem'] = vote['party'].str.contains('민주당',      case=False, na=False)
    vote['is_con'] = vote['party'].str.contains('통합당|한국당', case=False, na=False)

    # ---------- A. STATION LEVEL ----------
    area4['is_early']  = area4['name'].str.contains('prevote', case=False, na=False)
    a3_to_a2           = dict(zip(area3['uid'], area3['area2']))
    area4['area2_uid'] = area4['area3'].map(a3_to_a2)
    area4['region']    = area4['area2_uid'].map(dict(zip(area2['uid'], area2['name'])))

    def categorize_metro(name):
        if not isinstance(name, str): return "Other"
        if '인천' in name or '계양' in name or '연수' in name: return "Incheon"
        if '수원' in name or '고양' in name or '성남' in name or '용인' in name: return "Gyeonggi"
        if '서울' in name: return "Seoul"
        return "Other"

    area4['metro_zone']   = area4['region'].apply(categorize_metro)
    area4['invalid_rate'] = area4['sum_invalid'] / area4['sum_vote'].replace(0, np.nan)

    dem_votes_a4 = vote[vote['is_dem']].groupby('area')['vote'].sum()
    con_votes_a4 = vote[vote['is_con']].groupby('area')['vote'].sum()
    area4['dem_votes'] = area4['uid'].map(dem_votes_a4).fillna(0)
    area4['con_votes'] = area4['uid'].map(con_votes_a4).fillna(0)

    # ---------- B. DONG LEVEL ----------
    # Special-category area3 rows (prevote stations, disabled/absentee, overseas)
    # are NOT geographic dong districts and have no census counterpart.
    # They are excluded from dong-level analysis; their votes are still tagged
    # as 'early' correctly via area4_early below.
    SPECIAL_AREA3_PATTERN = r'prevote|ship|disabled|abroad'
    area3_special_uids = set(
        area3[area3['name'].str.contains(SPECIAL_AREA3_PATTERN, case=False, na=False)]['uid']
    )
    area3_geo  = area3[~area3['uid'].isin(area3_special_uids)].copy()
    area4_early = area4[area4['name'].str.contains('prevote', case=False, na=False)]['uid']

    vote_dong = vote.copy()
    vote_dong['vote_type'] = 'same_day'
    vote_dong.loc[
        vote_dong['area'].isin(area3_special_uids) | vote_dong['area'].isin(area4_early),
        'vote_type'
    ] = 'early'

    a4_to_a3 = dict(zip(area4['uid'], area4['area3']))
    vote_dong['area3_uid'] = vote_dong['area'].map(lambda x: a4_to_a3.get(x, x))

    # Drop rows whose resolved area3 is a special (non-geographic) category
    vote_dong = vote_dong[~vote_dong['area3_uid'].isin(area3_special_uids)]

    total_votes = (vote_dong.groupby(['area3_uid', 'vote_type'])['vote']
                            .sum().unstack(fill_value=0).reset_index()
                            .rename(columns={'early': 'total_early', 'same_day': 'total_same_day'}))

    party_agg = (vote_dong[vote_dong['is_dem']].groupby(['area3_uid', 'vote_type'])['vote']
                         .sum().unstack(fill_value=0).reset_index()
                         .rename(columns={'early': 'early_votes', 'same_day': 'same_day_votes'}))

    df_dong = pd.merge(party_agg, total_votes, on='area3_uid')

    # Attach area2 name and extract normalised match keys (geographic dongs only)
    area3_geo['area2_name'] = area3_geo['area2'].map(dict(zip(area2['uid'], area2['name'])))

    def get_election_sgg_candidates(area2_name: str):
        """
        Extract si/gun/gu match-key candidates from the area2 name.

        Two area2 naming conventions exist in the DB:
          A) Multi-region: "label_actual_municipality"
             e.g. '홍성군횡성군영월군평창군_홍천군' -> actual = '홍천군' -> ['홍천']
             e.g. '동해시태백시삼척시정선군_동해시' -> actual = '동해시' -> ['동해']
             The label before '_' is just the constituency group name and must
             NOT be parsed as an sgg key.
          B) Single-region: "city[갑을병정무]"
             e.g. '고양시갑' -> cleaned = '고양시' -> ['고양']
             e.g. '수원시장안구' -> ['장안', '수원']
        """
        if not isinstance(area2_name, str):
            return []
        if '_' in area2_name:
            # Convention A: the actual municipality is after the underscore
            actual = area2_name.split('_', 1)[1]          # e.g. '홍천군'
            return normalize_sigungu(actual)
        else:
            # Convention B: strip 갑/을/병/정/무 suffix then normalise
            cleaned = re.sub(r'[갑을병정무]$', '', area2_name).strip()
            return normalize_sigungu(cleaned)

    def get_election_dong_norm(area3_name: str) -> str:
        """
        Normalise an area3 district name to a comparable dong-level key.
        area3.name examples: '청운효자동', '역삼1동'
        (Special-category names are already excluded via area3_geo.)
        """
        if not isinstance(area3_name, str):
            return ""
        return normalize_dong_name(area3_name)

    # Province tag: map each area3_geo row to its short province name.
    # area1 is already loaded above — no second DB connection needed.
    # area1.name values are already the short forms: '경기', '강원', …
    a1_uid_to_name = dict(zip(area1['uid'], area1['name']))
    a2_uid_to_prov = {r['uid']: a1_uid_to_name.get(r['area1'], '')
                      for _, r in area2.iterrows()}
    area3_geo['province_tag'] = area3_geo['area2'].map(a2_uid_to_prov).fillna("")

    area3_geo['sgg_candidates'] = area3_geo['area2_name'].apply(get_election_sgg_candidates)
    area3_geo['primary_sgg']    = area3_geo['sgg_candidates'].apply(lambda x: x[0] if x else "")
    area3_geo['dong_norm']      = area3_geo['name'].apply(get_election_dong_norm)

    df_dong = pd.merge(
        df_dong,
        area3_geo[['uid', 'sgg_candidates', 'primary_sgg', 'dong_norm',
                   'province_tag', 'sum_people', 'sum_vote']],
        left_on='area3_uid', right_on='uid'
    )

    conn.close()
    return df_dong, area4


# ==========================================
# 3. MULTI-PASS DISTRICT MATCHER
# ==========================================

def merge_election_with_census(df_election: pd.DataFrame,
                                df_census:   pd.DataFrame) -> pd.DataFrame:
    """
    Three-pass merge that progressively relaxes constraints:

    Pass 1 – Exact:   primary_sgg (election) == primary_sgg (census)
                      AND dong_norm (election) == dong_norm (census)

    Pass 2 – Fallback sgg: try every alternative sgg candidate from the
                      election side against the census primary_sgg, keeping
                      dong_norm strict.

    Pass 3 – Fuzzy dong: for remaining unmatched rows, attempt a fuzzy
                      dong-name match (cutoff=0.82) within the same sgg bucket.

    Returns a merged DataFrame with a 'match_pass' column recording which
    pass produced each match, and prints a detailed reconciliation report.
    """
    if df_census.empty:
        print("[!] Census DataFrame is empty – skipping demographic merge.")
        return df_election.copy()

    # ── Census lookups ────────────────────────────────────────────────────────
    # (sgg, dong_norm)      -> propensity   [Passes 1-3]
    # (province, dong_norm) -> propensity   [Pass 5 – handles cases where the
    #   census 행정구역 omits a city-level token so the sgg index can't reach
    #   the city name, e.g. 고양시 dongs indexed only under '덕양구')
    census_lookup   = {}   # (sgg, dong_norm) -> propensity
    census_by_sgg   = {}   # sgg -> [dong_norm, …]  for fuzzy (Pass 3)
    prov_dong_map   = {}   # (province, dong_norm) -> propensity
    prov_dong_sum   = {}   # (province, dong_norm) -> propensity sum
    prov_dong_cnt   = {}   # (province, dong_norm) -> count
    dong_frequency    = {}   # dong_norm -> count across ALL census rows
    dong_unique_map   = {}   # dong_norm -> propensity  (only if globally unique)
    prov_sgg_dong_sum  = {}  # (province, sgg, dong_norm) -> propensity sum
    prov_sgg_dong_cnt  = {}  # same key -> count (for mean)

    PROV_FULL = {
        '서울특별시': '서울', '부산광역시': '부산', '대구광역시': '대구',
        '인천광역시': '인천', '광주광역시': '광주', '대전광역시': '대전',
        '울산광역시': '울산', '세종특별자치시': '세종',
        '경기도': '경기', '강원도': '강원', '충청북도': '충북',
        '충청남도': '충남', '전라북도': '전북', '전라남도': '전남',
        '경상북도': '경북', '경상남도': '경남', '제주특별자치도': '제주',
    }

    for _, row in df_census.iterrows():
        prop  = row['demographic_propensity']
        dnorm = row['dong_norm']
        # sgg-level index (Passes 1-3)
        for sgg in row['sgg_candidates']:
            census_lookup[(sgg, dnorm)] = prop
            census_by_sgg.setdefault(sgg, []).append(dnorm)
        # Province-level index (Pass 5)
        raw_name  = row.get('dong_raw', '')
        first_tok = raw_name.split()[0] if isinstance(raw_name, str) and raw_name.split() else ''
        prov      = PROV_FULL.get(first_tok, first_tok)
        pkey      = (prov, dnorm)
        prov_dong_sum[pkey] = prov_dong_sum.get(pkey, 0.0) + prop
        prov_dong_cnt[pkey] = prov_dong_cnt.get(pkey, 0) + 1
        # Province+sgg+dong triple index (Pass 6)
        # Use sum/count so that multiple sub-dongs (창1동…창5동) that normalize
        # to the same key get averaged rather than excluded by a freq filter.
        for sgg in row['sgg_candidates']:
            tkey = (prov, sgg, dnorm)
            prov_sgg_dong_sum[tkey] = prov_sgg_dong_sum.get(tkey, 0.0) + prop
            prov_sgg_dong_cnt[tkey] = prov_sgg_dong_cnt.get(tkey, 0) + 1
        # Global dong uniqueness (Pass 4)
        dong_frequency[dnorm] = dong_frequency.get(dnorm, 0) + 1
        dong_unique_map[dnorm] = prop

    # Prune ambiguous entries
    # Mean across sub-dongs normalizing to the same (province, dong) key.
    # e.g. 방림1동+방림2동 → ('광주','방림동'); mean propensity is correct
    # for election rows that cover the combined dong.
    prov_dong_map      = {k: v / prov_dong_cnt[k]
                          for k, v in prov_dong_sum.items()}
    prov_sgg_dong_map  = {k: v / prov_sgg_dong_cnt[k]
                          for k, v in prov_sgg_dong_sum.items()}
    dong_unique_map    = {d: p for d, p in dong_unique_map.items()
                          if dong_frequency[d] == 1}

    results = []
    stats   = {'pass1': 0, 'pass2': 0, 'pass3': 0, 'pass4': 0, 'pass5': 0, 'pass6': 0, 'pass7': 0, 'unmatched': 0}

    for _, row in df_election.iterrows():
        propensity  = None
        match_pass  = None
        dong_key    = row['dong_norm']
        sgg_cands   = row['sgg_candidates'] if isinstance(row['sgg_candidates'], list) else [row['primary_sgg']]

        # Pass 1 – primary sgg, exact dong
        key1 = (row['primary_sgg'], dong_key)
        if key1 in census_lookup:
            propensity = census_lookup[key1]
            match_pass = 'pass1'

        # Pass 2 – try alternate sgg keys, exact dong
        if propensity is None:
            for sgg in sgg_cands[1:]:
                key2 = (sgg, dong_key)
                if key2 in census_lookup:
                    propensity = census_lookup[key2]
                    match_pass = 'pass2'
                    break

        # Pass 3 – fuzzy dong match within any known sgg bucket
        if propensity is None:
            for sgg in sgg_cands:
                pool = census_by_sgg.get(sgg, [])
                if not pool:
                    continue
                matches = get_close_matches(dong_key, pool, n=1, cutoff=0.82)
                if matches:
                    candidate_key = (sgg, matches[0])
                    if candidate_key in census_lookup:
                        propensity = census_lookup[candidate_key]
                        match_pass = 'pass3'
                        break

        # Pass 4 – dong-unique fallback (no sgg constraint)
        if propensity is None:
            if dong_key in dong_unique_map:
                propensity = dong_unique_map[dong_key]
                match_pass = 'pass4'

        # Pass 5 – province + dong
        if propensity is None:
            prov = row.get('province_tag', '')
            if prov:
                pkey = (prov, dong_key)
                if pkey in prov_dong_map:
                    propensity = prov_dong_map[pkey]
                    match_pass = 'pass5'

        # Pass 6 – province + sgg + dong (mean propensity across sub-dongs)
        if propensity is None:
            prov = row.get('province_tag', '')
            if prov:
                for sgg in sgg_cands:
                    tkey = (prov, sgg, dong_key)
                    if tkey in prov_sgg_dong_map:
                        propensity = prov_sgg_dong_map[tkey]
                        match_pass = 'pass6'
                        break

        # Pass 7 – non-numeric dot component lookup
        # Handles merged dongs like '용담·명암·산성동' where the census stores
        # each component (용담동, 명암동, 산성동) as a separate row.
        # Only fires when the dong_key contains a middle-dot between non-digits.
        if propensity is None and '·' in dong_key:
            raw_components = dong_key.split('·')
            prop_hits = []
            for comp in raw_components:
                # Add 동 suffix if component has no administrative ending
                if not re.search(r'(동|읍|면|가)$', comp):
                    comp = comp + '동'
                comp_norm = normalize_dong_name(comp)
                prov = row.get('province_tag', '')
                # Try sgg-level first, then province+sgg triple
                for sgg in sgg_cands:
                    k = (sgg, comp_norm)
                    if k in census_lookup:
                        prop_hits.append(census_lookup[k])
                        break
                    if prov:
                        tk = (prov, sgg, comp_norm)
                        if tk in prov_sgg_dong_map:
                            prop_hits.append(prov_sgg_dong_map[tk])
                            break
            if prop_hits:
                propensity = sum(prop_hits) / len(prop_hits)
                match_pass = 'pass7'

        if propensity is None:
            stats['unmatched'] += 1
        else:
            stats[match_pass] += 1

        row_dict = row.to_dict()
        row_dict['demographic_propensity'] = propensity
        row_dict['match_pass']             = match_pass
        results.append(row_dict)

    df_out = pd.DataFrame(results)
    df_out = df_out[df_out['demographic_propensity'].notna()].copy()

    total    = len(df_election)
    matched  = total - stats['unmatched']
    print(f"\n  ┌─ Matching Report ──────────────────────────")
    print(f"  │  Total election districts : {total:>6,}")
    print(f"  │  Matched (Pass 1 – exact) : {stats['pass1']:>6,}")
    print(f"  │  Matched (Pass 2 – alt sgg): {stats['pass2']:>6,}")
    print(f"  │  Matched (Pass 3 – fuzzy) : {stats['pass3']:>6,}")
    print(f"  │  Matched (Pass 4 – dong unique): {stats['pass4']:>4,}")
    print(f"  │  Matched (Pass 5 – province+dong): {stats['pass5']:>3,}")
    print(f"  │  Matched (Pass 6 – prov+sgg+dong): {stats['pass6']:>3,}")
    print(f"  │  Matched (Pass 7 – dot components): {stats['pass7']:>3,}")
    print(f"  │  Unmatched                : {stats['unmatched']:>6,}  ({stats['unmatched']/total*100:.1f}%)")

    # ── Diagnostic: for still-unmatched rows, show whether dong_key exists
    # anywhere in census (sgg mismatch) or is absent entirely (census gap)
    if stats['unmatched'] > 0:
        census_dongs_any_sgg = {k[1] for k in census_lookup}
        unmatched_rows = [r for r in results if r['match_pass'] is None]
        print(f"\n  [Unmatched districts (first 20)]")
        for r in unmatched_rows[:20]:
            in_census = r['dong_norm'] in census_dongs_any_sgg
            flag = "sgg-mismatch" if in_census else "NOT-IN-CENSUS"
            print(f"    [{flag}]  sgg_cands={r['sgg_candidates']}  dong='{r['dong_norm']}'")
    print(f"  └────────────────────────────────────────────")



    return df_out


# ==========================================
# 4. THE FORENSICS ENGINE
# ==========================================

def run_comprehensive_forensics_suite(df_dong, df_station, df_demo):
    print(f"\n--- [3/3] Executing Comprehensive Forensics Suite ---")

    # Merge census
    df_merged = merge_election_with_census(df_dong, df_demo)

    # Filter noise
    df_merged = df_merged[
        (df_merged['total_early']    > 50) &
        (df_merged['total_same_day'] > 50)
    ].copy()

    # Derived metrics
    df_merged['early_pct']        = df_merged['early_votes']   / df_merged['total_early']
    df_merged['same_day_pct']     = df_merged['same_day_votes'] / df_merged['total_same_day']
    df_merged['raw_vote_gap']     = df_merged['early_pct'] - df_merged['same_day_pct']
    df_merged['weighted_vote_gap']= df_merged['raw_vote_gap'] / df_merged['demographic_propensity']
    df_merged['vote_share']       = ((df_merged['early_votes'] + df_merged['same_day_votes']) /
                                     (df_merged['total_early'] + df_merged['total_same_day']))
    df_merged['turnout']          = df_merged['sum_vote'] / df_merged['sum_people'].replace(0, np.nan)

    print("\n" + "="*50)
    print("   PART 1: STANDARD FORENSICS (DIGITS & DISPERSION)")
    print("="*50)

    # Test A: 2BL
    valid_votes   = df_merged['early_votes'].astype(str)
    second_digits = valid_votes.str[1].astype(int)
    obs_2bl  = second_digits.value_counts().reindex(range(10), fill_value=0).sort_index()
    exp_2bl  = [sum(math.log10(1 + (1 / (10*k + d))) for k in range(1, 10)) * len(valid_votes)
                for d in range(10)]
    chi_2bl, p_2bl = chisquare(f_obs=obs_2bl, f_exp=exp_2bl)
    print(f"[*] 2BL Test -> Chi-Square: {chi_2bl:.2f} | P-Value: {p_2bl:.4f}")

    # Test B: Last Digit
    last_digits = valid_votes.str[-1].astype(int)
    obs_ld  = last_digits.value_counts().reindex(range(10), fill_value=0).sort_index()
    exp_ld  = [len(valid_votes) / 10] * 10
    chi_ld, p_ld = chisquare(f_obs=obs_ld, f_exp=exp_ld)
    print(f"[*] Last Digit Uniformity -> Chi-Square: {chi_ld:.2f} | P-Value: {p_ld:.4f}")

    # Test C: Variance
    variances = df_merged.groupby('primary_sgg')['vote_share'].std().dropna() * 100
    print(f"[*] Variance Test -> Avg StDev between neighbourhoods: {variances.mean():.2f}%")

    print("\n" + "="*50)
    print("   PART 2: TESTING CONSPIRACY THEORIES")
    print("="*50)

    # Test D: Constant Gap
    mean_gap, std_gap = df_merged['raw_vote_gap'].mean(), df_merged['raw_vote_gap'].std()
    print(f"[*] 'Constant Gap' Theory -> Mean Gap: +{mean_gap*100:.2f}% | StDev: {std_gap*100:.2f}%")

    # Test E: Algorithmic Ratio
    corr, _ = pearsonr(df_merged['same_day_pct'], df_merged['early_pct'])
    r2 = corr ** 2
    print(f"[*] 'Algorithmic Ratio' Theory -> R²: {r2:.4f}")

    # Test F: 63:36 Metro Aggregate
    early_stations = df_station[df_station['is_early'] == True].copy()
    metro_agg = early_stations.groupby('metro_zone')[['dem_votes', 'con_votes']].sum()
    metro_agg['total']     = metro_agg['dem_votes'] + metro_agg['con_votes']
    metro_agg['dem_ratio'] = metro_agg['dem_votes'] / metro_agg['total'] * 100
    print(f"\n[*] '63:36' Theory -> Macroscopic Aggregates:")
    for zone, row in metro_agg.iterrows():
        if zone != "Other" and row['total'] > 0:
            print(f"    - {zone}: Dem {row['dem_ratio']:.2f}% | Con {100-row['dem_ratio']:.2f}%")
    micro_std = early_stations['dem_votes'].div(
        (early_stations['dem_votes'] + early_stations['con_votes']).replace(0, np.nan)
    ).std() * 100
    print(f"    - Microscopic Station Variance (StDev): {micro_std:.2f}%")

    # Test G: Invalid Vote Anomaly
    early_clean = early_stations.dropna(subset=['invalid_rate'])
    early_clean = early_clean.copy()
    early_clean['micro_dem_ratio'] = (
        early_clean['dem_votes'] /
        (early_clean['dem_votes'] + early_clean['con_votes']).replace(0, np.nan)
    ) * 100
    early_clean = early_clean.dropna(subset=['micro_dem_ratio'])
    inv_corr, inv_pval = pearsonr(early_clean['invalid_rate'], early_clean['micro_dem_ratio'])
    print(f"\n[*] 'Invalid Vote' Theory -> Correlation: {inv_corr:.4f} (P-Val: {inv_pval:.4f})")

    return df_merged, obs_ld, exp_ld, variances, r2


# ==========================================
# 5. MEGA DASHBOARD
# ==========================================

def plot_mega_dashboard(df, obs_ld, exp_ld, variances, r2):
    print("\nGenerating Mega Dashboard Image...")
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Comprehensive Election Forensics Dashboard",
                 fontsize=20, fontweight='bold', y=0.98)

    axes[0, 0].hist(df['raw_vote_gap'] * 100, bins=50, color='red', alpha=0.6, edgecolor='black')
    axes[0, 0].axvline(df['raw_vote_gap'].mean() * 100, color='black', linestyle='dashed', linewidth=2)
    axes[0, 0].set_title('1. Raw Early vs Same-Day Gap')
    axes[0, 0].set_xlabel('Difference (%)')

    axes[0, 1].hist(df['weighted_vote_gap'] * 100, bins=50, color='green', alpha=0.6, edgecolor='black')
    axes[0, 1].axvline(df['weighted_vote_gap'].mean() * 100, color='black', linestyle='dashed', linewidth=2)
    axes[0, 1].set_title('2. Gap Adjusted by Neighbourhood Demographics')
    axes[0, 1].set_xlabel('Weighted Difference Score')

    min_v = min(df['same_day_pct'].min(), df['early_pct'].min()) * 100
    max_v = max(df['same_day_pct'].max(), df['early_pct'].max()) * 100
    axes[1, 0].scatter(df['same_day_pct']*100, df['early_pct']*100,
                       alpha=0.3, color='blue', edgecolor='k', s=20)
    axes[1, 0].plot([min_v, max_v], [min_v, max_v], 'r--', label='Perfect 1:1 Correlation')
    axes[1, 0].set_title(f'3. Algorithmic Ratio Check (R² = {r2:.4f})')
    axes[1, 0].set_xlabel('Same-Day Democratic Share (%)')
    axes[1, 0].set_ylabel('Early Democratic Share (%)')
    axes[1, 0].legend()

    fp = df[(df['turnout'] <= 1.0) & (df['turnout'] > 0)]
    hb = axes[1, 1].hexbin(fp['turnout'] * 100, fp['vote_share'] * 100,
                            gridsize=30, cmap='inferno', mincnt=1)
    fig.colorbar(hb, ax=axes[1, 1], label='Number of Districts')
    axes[1, 1].set_title('4. Election Fingerprint (Turnout vs Vote Share)')
    axes[1, 1].set_xlabel('Voter Turnout (%)')
    axes[1, 1].set_ylabel('Democratic Vote Share (%)')

    axes[2, 0].bar(range(10), obs_ld, color='salmon', alpha=0.8, edgecolor='black', label='Observed')
    axes[2, 0].plot(range(10), exp_ld, color='black', linestyle='dashed', linewidth=2, label='Expected Uniform')
    axes[2, 0].set_title('5. Last Digit Uniformity Test')
    axes[2, 0].set_xlabel('Last Digit (0-9)')
    axes[2, 0].set_xticks(range(10))
    axes[2, 0].legend()

    axes[2, 1].hist(variances, bins=30, color='purple', alpha=0.7, edgecolor='black')
    axes[2, 1].set_title('6. Dispersion Test (Vote Share Variance by City)')
    axes[2, 1].set_xlabel('Standard Deviation of Vote Share (%)')
    axes[2, 1].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('mega_forensics_dashboard.png', dpi=300)
    print("Saved 'mega_forensics_dashboard.png'")


# ==========================================
# EXECUTION
# ==========================================

if __name__ == "__main__":
    df_demo              = fetch_kosis_demographics_csv()
    df_dong, df_station  = fetch_election_data()

    if not df_dong.empty and not df_station.empty:
        df_merged, obs_ld, exp_ld, variances, r2 = run_comprehensive_forensics_suite(
            df_dong, df_station, df_demo
        )
        plot_mega_dashboard(df_merged, obs_ld, exp_ld, variances, r2)

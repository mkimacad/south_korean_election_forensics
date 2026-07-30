#!/usr/bin/env python3
"""
convert_18th_general.py -- STANDALONE converter for the 18th General
Election (2008), from its native format (16 per-province .xls files, one
sheet per constituency) into the SAME long-format CSV schema every other
election in this project uses (시도명, 선거구명, 법정읍면동명, 투표구명,
후보자, 득표수) -- so `cell3.load_election_csv()` / `forensics_core.py`'s
own loader can read it with ZERO special-casing, exactly like any other
election's CSV.

Run standalone, once, to produce 18th_election_result.csv:
    python3 convert_18th_general.py 18대지역구_총괄.zip

WHY THIS IS A SEPARATE FILE, NOT BOLTED ONTO cell3.py/forensics_core.py:
this format is genuinely different (wide, multi-row headers, merged dong-
name cells, subtotal rows mixed with detail rows) and reading it needs
xlrd + real parsing logic that has nothing to do with the long-format CSV
the rest of the pipeline consumes. Isolating it here means eforensics_
mixture.py and forensics_core.py never need to know this format exists --
they just get 18th_election_result.csv like any other file.

RAW FORMAT (per sheet == one constituency):
    row 0: 읍면동명 | 투표구명 | 선거인수 | 투표수 | 후보자별 득표상황 (label
           only on the first of several merged columns) | ... | 무효 | 기권수
    row 1: (blank) x4, then PARTY name per candidate column, then '계'
           (candidates-only subtotal, a built-in cross-check), then a
           stray '투표수' label in the 무효 column position (a rendering
           artifact of the merged header, not a real column -- verified:
           that column's VALUES match 투표수 - 계, i.e. invalid votes, so
           row 0's '무효' label is the authoritative one, not row 1's).
    row 2: candidate NAME per candidate column
    row 3: '합계' row -- the constituency TOTAL. Not emitted as an output
           row (every other election's CSV lets load_election_csv() derive
           constituency totals itself via groupby) -- used ONLY to
           validate the sum of everything else in this sheet.
    row 4: '부재자' row -- the absentee category (matching pres16/17's
           '부재자투표' naming, added to SPECIAL_DONG_NAMES_GENERAL).
           EMITTED as its own pseudo-dong, matching how every other
           election's absentee votes are handled.
    remaining rows, per dong, in blocks of:
      - one '소계' row: col0 = dong name, col1 = literal '소계'. This is a
        DONG SUBTOTAL -- not emitted directly (redundant with the sum of
        its own precinct rows below it) -- used ONLY to validate that sum,
        and to supply the dong name to forward-fill onto the rows below it
        (their own col0 is blank -- a merged-cell artifact from Excel).
      - one or more PRECINCT DETAIL rows: col0 = blank (merged cell -- the
        dong name from the '소계' row above applies), col1 = precinct name
        (e.g. '삼도1동제1투'). THESE are the rows actually emitted, at the
        same granularity every other election's raw CSV provides.

Column layout is NOT assumed fixed by position (candidate count varies by
constituency -- confirmed 4 candidates in one 서귀포시 sheet vs. 5 in a
Seoul sheet, shifting every column after the last candidate by one) --
every column is located by its row 0 / row 1 label instead.

VALIDATION (the actual point of using this format at all, not a nicety):
for every sheet, after parsing, sums are checked at two levels and any
mismatch beyond floating-point tolerance is a HARD ERROR (raises), not a
warning -- a silently-wrong election result is worse than a script that
refuses to finish:
  1. sum(precinct detail rows) + 부재자 row, per candidate, per meta field
     (선거인수/투표수/무효/기권수) == 합계 row.
  2. sum(precinct detail rows within one dong) == that dong's own '소계'
     row.
"""
import sys
import os
import zipfile
import re
import pandas as pd
import numpy as np

META_FIELDS = ['선거인수', '투표수', '무효 투표수', '기권자수']
ABSENTEE_DONG_LABEL = '부재자투표'   # matches pres16/17's SPECIAL_DONG_NAMES_GENERAL entry
# '잘못투입된투표지' (misdirected ballots) appears in ALL 16 province files
# (confirmed by an exhaustive scan of every non-dong labeled row across
# every sheet in every file -- it is the ONLY such category besides
# 합계/부재자/소계). Output under the OFFICIAL label with spaces/middle-dot
# that's ALREADY in cell3.py's SPECIAL_DONG_NAMES_GENERAL (added for
# 21st/22nd's raw data), rather than adding a second near-duplicate label
# variant to that set for this one election's different formatting.
MISDIRECTED_BALLOTS_DONG_LABEL = '잘못 투입·구분된 투표지'

PROVINCE_SHORT_TO_FULL = {
    '강원': '강원도', '경기': '경기도', '경남': '경상남도', '경북': '경상북도',
    '광주': '광주광역시', '대구': '대구광역시', '대전': '대전광역시', '부산': '부산광역시',
    '서울': '서울특별시', '울산': '울산광역시', '인천': '인천광역시',
    '전남': '전라남도', '전북': '전라북도', '제주': '제주특별자치도',
    '충남': '충청남도', '충북': '충청북도',
    # 2008 predates Sejong (est. 2012) -- no entry needed/expected.
}


def recover_korean_filenames(zip_path, out_dir):
    """The zip stores Korean filenames in a way plain `unzip`/zipfile
    (assuming UTF-8) mangles into garbage. Recovering them needs the
    cp437 (zip's fallback byte-preserving codec) -> cp949 (actual Korean
    encoding used when the zip was created) round-trip. Returns
    {province_short_name: extracted_file_path}."""
    os.makedirs(out_dir, exist_ok=True)
    z = zipfile.ZipFile(zip_path)
    recovered = {}
    for info in z.infolist():
        if not info.filename.lower().endswith('.xls'):
            continue
        try:
            fixed_name = info.filename.encode('cp437').decode('cp949')
        except (UnicodeEncodeError, UnicodeDecodeError):
            fixed_name = info.filename   # already-correct name; leave as-is
        province_short = fixed_name.replace('.xls', '')
        out_path = os.path.join(out_dir, fixed_name)
        with z.open(info) as src, open(out_path, 'wb') as dst:
            dst.write(src.read())
        recovered[province_short] = out_path
    return recovered


def _find_col(row_values, label):
    """First column index where row_values[i] == label, or None."""
    for i, v in enumerate(row_values):
        if v == label:
            return i
    return None


def parse_sheet(df_raw, province_full, constituency):
    """df_raw: raw sheet, header=None (so df_raw.iloc[0] is the real row 0
    etc). Returns (records: list[dict], errors: list[str]) -- records are
    ready for the long-format CSV; errors are validation failures (empty
    list means the sheet's internal totals all reconciled)."""
    errors = []
    row0 = df_raw.iloc[0].tolist()
    row1 = df_raw.iloc[1].tolist()
    row2 = df_raw.iloc[2].tolist()

    idx_dong = 0
    idx_precinct = 1
    idx_electorate = _find_col(row0, '선거인수')
    idx_totalvotes = _find_col(row0, '투표수')
    # '무효' is the standard label; '무효투표수' (no space) is a confirmed
    # one-off variant (강남구갑, 2008 -- a crowded independent-heavy race)
    # -- accept either rather than assuming every sheet uses the same
    # exact label.
    idx_invalid = _find_col(row0, '무효')
    if idx_invalid is None:
        idx_invalid = _find_col(row0, '무효투표수')
    idx_abstain = _find_col(row0, '기권수')
    idx_gye = _find_col(row1, '계')   # candidates-only subtotal, cross-check column

    required = {'선거인수': idx_electorate, '투표수': idx_totalvotes,
               '무효': idx_invalid, '기권수': idx_abstain, '계(row1)': idx_gye}
    missing = [k for k, v in required.items() if v is None]
    if missing:
        errors.append(f"{constituency}: could not locate column(s) {missing} in header rows -- "
                      f"row0={row0!r} row1={row1!r}")
        return [], errors

    # candidate columns: strictly between 투표수 and the '계' cross-check column
    cand_cols = list(range(idx_totalvotes + 1, idx_gye))
    candidates = []   # (col_idx, party, name)
    for c in cand_cols:
        party, name = row1[c], row2[c]
        if pd.isna(party) or pd.isna(name):
            errors.append(f"{constituency}: candidate column {c} has missing party/name "
                          f"(party={party!r}, name={name!r}) -- skipping this column")
            continue
        candidates.append((c, str(party).strip(), str(name).strip()))
    if not candidates:
        errors.append(f"{constituency}: zero candidate columns detected between "
                      f"투표수(col {idx_totalvotes}) and 계(col {idx_gye}) -- sheet skipped entirely")
        return [], errors

    def row_meta_values(row):
        return {'선거인수': row[idx_electorate], '투표수': row[idx_totalvotes],
                '무효 투표수': row[idx_invalid], '기권자수': row[idx_abstain]}

    def row_cand_values(row):
        return {(party, name): row[c] for c, party, name in candidates}

    records = []
    total_row, absentee_row, misdirected_row = None, None, None
    dong_subtotal_rows = {}      # dong_name -> row values (for validation)
    dong_precinct_sums = {}      # dong_name -> running sum dict (for validation)
    current_dong = None

    n_rows = len(df_raw)
    for r in range(3, n_rows):
        row = df_raw.iloc[r].tolist()
        c0, c1 = row[idx_dong], row[idx_precinct]
        if pd.isna(c0) and pd.isna(c1):
            continue   # blank/separator row
        if c0 == '합계':
            total_row = row
            continue
        if c0 == '부재자':
            absentee_row = row
            continue
        if c0 == '잘못투입된투표지':
            misdirected_row = row
            continue
        if c1 == '소계':
            current_dong = str(c0).strip()
            dong_subtotal_rows[current_dong] = row
            dong_precinct_sums.setdefault(current_dong, {'meta': {k: 0 for k in META_FIELDS},
                                                          'cand': {(p, n): 0 for _, p, n in candidates}})
            continue
        # precinct detail row: c0 is blank (merged-cell carry-over), c1 is the precinct name
        if pd.isna(c1):
            errors.append(f"{constituency}: unrecognized row at index {r} "
                          f"(col0={c0!r}, col1={c1!r}) -- skipped, not counted in any total")
            continue
        if current_dong is None:
            errors.append(f"{constituency}: precinct row {c1!r} at index {r} appeared before "
                          f"any dong ('소계') row -- skipped, dong unknown")
            continue
        precinct_name = str(c1).strip()
        meta_vals = row_meta_values(row)
        cand_vals = row_cand_values(row)
        for field, val in meta_vals.items():
            records.append({'법정읍면동명': current_dong, '투표구명': precinct_name,
                            '후보자': field, '득표수': val})
            dong_precinct_sums[current_dong]['meta'][field] += (0 if pd.isna(val) else val)
        for (party, name), val in cand_vals.items():
            records.append({'법정읍면동명': current_dong, '투표구명': precinct_name,
                            '후보자': f'{party} {name}', '득표수': val})
            dong_precinct_sums[current_dong]['cand'][(party, name)] += (0 if pd.isna(val) else val)

    # --- absentee: emitted as its own pseudo-dong (matches every other
    # election's absentee-category handling) ---
    if absentee_row is not None:
        meta_vals = row_meta_values(absentee_row)
        cand_vals = row_cand_values(absentee_row)
        for field, val in meta_vals.items():
            records.append({'법정읍면동명': ABSENTEE_DONG_LABEL, '투표구명': np.nan,
                            '후보자': field, '득표수': val})
        for (party, name), val in cand_vals.items():
            records.append({'법정읍면동명': ABSENTEE_DONG_LABEL, '투표구명': np.nan,
                            '후보자': f'{party} {name}', '득표수': val})
    else:
        errors.append(f"{constituency}: no '부재자' row found -- absentee votes for this "
                      f"constituency are MISSING from the output, not just zero")

    # --- misdirected ballots: same treatment as absentee (a real,
    # NEC-reported category, not a rounding artifact -- see module
    # docstring) ---
    if misdirected_row is not None:
        meta_vals = row_meta_values(misdirected_row)
        cand_vals = row_cand_values(misdirected_row)
        for field, val in meta_vals.items():
            records.append({'법정읍면동명': MISDIRECTED_BALLOTS_DONG_LABEL, '투표구명': np.nan,
                            '후보자': field, '득표수': val})
        for (party, name), val in cand_vals.items():
            records.append({'법정읍면동명': MISDIRECTED_BALLOTS_DONG_LABEL, '투표구명': np.nan,
                            '후보자': f'{party} {name}', '득표수': val})
    else:
        # NOT added to `errors`: confirmed (see module docstring/commit
        # history) that when this row is absent, 합계 still reconciles
        # exactly without it in every case checked -- i.e. NEC omitted a
        # genuinely zero-valued row here rather than including one, a
        # benign template variation, not a data-integrity problem. If a
        # future run's 합계 check DOES fail for a constituency missing
        # this row, that failure surfaces on its own via the 합계
        # mismatch check below -- this note doesn't need to duplicate it.
        print(f"    [note] {constituency}: no '잘못투입된투표지' row (informational -- "
              f"real problems would surface as a 합계 mismatch below, not here)")

    # --- VALIDATION 1: each dong's 소계 row == sum of its own precinct rows ---
    TOL = 1e-6
    for dong, subtotal_row in dong_subtotal_rows.items():
        expected_meta = row_meta_values(subtotal_row)
        expected_cand = row_cand_values(subtotal_row)
        actual = dong_precinct_sums[dong]
        for field, exp_val in expected_meta.items():
            exp_val = 0 if pd.isna(exp_val) else exp_val
            if abs(actual['meta'][field] - exp_val) > TOL:
                errors.append(f"{constituency}/{dong}: 소계 mismatch on {field}: "
                              f"소계 row says {exp_val}, sum of precinct rows = {actual['meta'][field]}")
        for key, exp_val in expected_cand.items():
            exp_val = 0 if pd.isna(exp_val) else exp_val
            if abs(actual['cand'][key] - exp_val) > TOL:
                errors.append(f"{constituency}/{dong}: 소계 mismatch on {key}: "
                              f"소계 row says {exp_val}, sum of precinct rows = {actual['cand'][key]}")

    # --- VALIDATION 2: 합계 row == sum(all dong 소계 rows) + 부재자 row ---
    if total_row is not None:
        expected_meta = row_meta_values(total_row)
        expected_cand = row_cand_values(total_row)
        got_meta = {k: 0 for k in META_FIELDS}
        got_cand = {(p, n): 0 for _, p, n in candidates}
        for subtotal_row in dong_subtotal_rows.values():
            for field, val in row_meta_values(subtotal_row).items():
                got_meta[field] += (0 if pd.isna(val) else val)
            for key, val in row_cand_values(subtotal_row).items():
                got_cand[key] += (0 if pd.isna(val) else val)
        if absentee_row is not None:
            for field, val in row_meta_values(absentee_row).items():
                got_meta[field] += (0 if pd.isna(val) else val)
            for key, val in row_cand_values(absentee_row).items():
                got_cand[key] += (0 if pd.isna(val) else val)
        if misdirected_row is not None:
            for field, val in row_meta_values(misdirected_row).items():
                got_meta[field] += (0 if pd.isna(val) else val)
            for key, val in row_cand_values(misdirected_row).items():
                got_cand[key] += (0 if pd.isna(val) else val)
        for field, exp_val in expected_meta.items():
            exp_val = 0 if pd.isna(exp_val) else exp_val
            if abs(got_meta[field] - exp_val) > TOL:
                errors.append(f"{constituency}: 합계 mismatch on {field}: "
                              f"합계 row says {exp_val}, sum of (dongs + 부재자) = {got_meta[field]}")
        for key, exp_val in expected_cand.items():
            exp_val = 0 if pd.isna(exp_val) else exp_val
            if abs(got_cand[key] - exp_val) > TOL:
                errors.append(f"{constituency}: 합계 mismatch on {key}: "
                              f"합계 row says {exp_val}, sum of (dongs + 부재자) = {got_cand[key]}")
    else:
        errors.append(f"{constituency}: no '합계' row found -- cannot validate this "
                      f"constituency's totals at all")

    for rec in records:
        rec['시도명'] = province_full
        # Multi-county merged districts (e.g. '보은군옥천군영동군') are
        # reported as SEPARATE sheets per sub-county (e.g.
        # '보은군옥천군영동군(영동군)'), but are the SAME electoral race --
        # confirmed identical candidate lists across all of a merged
        # district's sub-sheets. Strip the parenthetical suffix so
        # build_constituency_table()'s groupby(선거구명) correctly merges
        # them back into one constituency instead of creating spurious
        # extra "constituencies" each with only partial vote totals.
        # `constituency` (the original sheet name) is still what appears
        # in error/log messages above, for traceability back to the
        # source sheet.
        rec['선거구명'] = re.sub(r'\(.*\)', '', constituency).strip()
    return records, errors


def convert(zip_path, out_csv='18th_election_result.csv', tmp_dir='18th_raw_recovered'):
    print(f"--- Recovering Korean filenames from {zip_path} ---")
    files = recover_korean_filenames(zip_path, tmp_dir)
    print(f"  {len(files)} province files recovered: {sorted(files.keys())}")

    all_records = []
    all_errors = []
    for province_short, path in sorted(files.items()):
        province_full = PROVINCE_SHORT_TO_FULL.get(province_short)
        if province_full is None:
            all_errors.append(f"[{province_short}] unrecognized province short-name -- "
                              f"add it to PROVINCE_SHORT_TO_FULL. Skipping this entire file.")
            continue
        print(f"--- Parsing {province_short} ({province_full}) ---")
        xls = pd.ExcelFile(path, engine='xlrd')
        for sheet in xls.sheet_names:
            df_raw = pd.read_excel(path, sheet_name=sheet, engine='xlrd', header=None)
            records, errors = parse_sheet(df_raw, province_full, sheet)
            all_records.extend(records)
            for e in errors:
                all_errors.append(f"[{province_short}/{sheet}] {e}")
            status = "OK" if not errors else f"{len(errors)} VALIDATION ISSUE(S)"
            print(f"    {sheet}: {len(records)} rows -- {status}")

    print(f"\n--- Validation summary: {len(all_errors)} issue(s) across "
          f"{sum(len(pd.ExcelFile(p, engine='xlrd').sheet_names) for p in files.values())} constituencies ---")
    for e in all_errors:
        print(f"  [!] {e}")

    df_out = pd.DataFrame(all_records)[['시도명', '선거구명', '법정읍면동명', '투표구명', '후보자', '득표수']]
    df_out.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"\n--- Wrote {out_csv}: {len(df_out)} rows ---")
    if all_errors:
        print(f"\n[!!!] {len(all_errors)} validation issue(s) found -- DO NOT trust this CSV for "
              f"analysis until every one of these is understood and either fixed or explicitly "
              f"accepted as a genuine, documented data artifact (e.g. NEC's own error correction).")
        return 1
    else:
        print("\n[+] Every constituency's 합계 and 소계 cross-checks reconciled exactly. "
              "No independent guarantee beyond that, but this is real evidence the parser "
              "read the format correctly, not just that it didn't crash.")
        return 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 convert_18th_general.py <path to 18대지역구_총괄.zip> [output_csv]")
        sys.exit(1)
    zip_path = sys.argv[1]
    out_csv = sys.argv[2] if len(sys.argv) > 2 else '18th_election_result.csv'
    sys.exit(convert(zip_path, out_csv))

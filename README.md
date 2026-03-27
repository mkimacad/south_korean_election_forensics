# southkorea_election
South Korean election forensics

==========

Code information:

The main forensic file (outdated, to be updated/simplified so that regressions are solely done in the regression file):

korean_election_forensics.ipynb

The main regression file: korean_election_regressions.ipynb

Required data files (./ as the working directory where the python or jupyter notebook files are at):

place processed_apt_data_*.csv files into ./output_data directory

place other csv data into ./: stations_normal_*.csv files, industry_employment_data.csv, men_employment_data.csv, non_participant_employment_data.csv, occupation_employment_data.csv, occupation_type_employment_data.csv, women_employment_data.csv and all *_census.csv and *_result.csv files.

==========

Regression results:

See comprehensive_regression_logs_*.txt files. The regression results used in the manuscript have no turnout covariate in democratic vote share regressions - this is the baseline we use.

==========

Manuscript (completely outdated, to be updated): south_korean_election_forensic.pdf

==========

The census data (csv files) are from https://www.data.go.kr/data/3033304/fileData.do (official Korean government census data), which directs to https://jumin.mois.go.kr/ageStatMonth.do .

The general elction results (csv files) are from official Korean government data, https://www.data.go.kr/data/15025527/fileData.do
(The presidental election results are from https://www.data.go.kr/data/15025528/fileData.do)

The actual monthly housing price data are from official Korean government data, https://www.data.go.kr/data/3050988/fileData.do, which directs to https://rt.molit.go.kr/pt/xls/xls.do?mobileAt=

(Additional data: 시군구별 경제활동인구조사 지역별고용조사 + GRDP data from KOSIS)

Polling station information data from http://data.go.kr/data/15000836/openapi.do

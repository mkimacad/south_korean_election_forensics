import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import re

# ==========================================
# CONFIGURATION
# ==========================================
SERVICE_KEY = ''
ELECTION_ID = '20240410'  
FETCH_MODE  = 'pre'    # 'normal' for Election Day, 'pre' for Early Voting
OUTPUT_FILE = f"stations_{FETCH_MODE}_{ELECTION_ID}.csv"

# ==========================================
# CRAWLER & NORMALIZATION
# ==========================================
def fetch_data():
    base_url = 'http://apis.data.go.kr/9760000/PolplcInfoInqireService2/'
    endpoint = 'getPolplcOtlnmapTrnsportInfoInqire' if FETCH_MODE == 'normal' else 'getPrePolplcOtlnmapTrnsportInfoInqire'
    
    provinces = ['서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', '대전광역시', '울산광역시', 
                 '세종특별자치시', '경기도', '강원특별자치도', '충청북도', '충청남도', '전북특별자치도', 
                 '전라남도', '경상북도', '경상남도', '제주특별자치도']
    
    all_stations = []
    
    for sd in provinces:
        print(f"[*] Fetching {FETCH_MODE} stations for {sd}...")
        page = 1
        
        while True:
            params = {
                'serviceKey': SERVICE_KEY, 
                'pageNo': str(page), 
                'numOfRows': '100',  # <-- FORCED TO 100 to match their backend cap
                'sgId': ELECTION_ID, 
                'sdName': sd
            }
            
            try:
                res = requests.get(base_url + endpoint, params=params, timeout=15)
                root = ET.fromstring(res.text)
                
                items = root.findall('.//item')
                for item in items:
                    all_stations.append({
                        'sdName': item.findtext('sdName'),
                        'wiwName': item.findtext('wiwName'),
                        'emdName': item.findtext('emdName')
                    })
                
                print(f"    -> Page {page}: Extracted {len(items)} stations")
                
                # If the server gives us less than 100 rows, we have finally reached the last page!
                if len(items) < 100:
                    break
                    
                page += 1
                time.sleep(0.1) # Safety delay
                
            except Exception as e:
                print(f"  [!] Error on Page {page}: {e}")
                break

    df = pd.DataFrame(all_stations)
    print(f"\n[!] Total raw stations collected nationally: {len(df)}")
    
    # Simple counting (you can re-apply your normalization strings here)
    counts = df.groupby(['sdName', 'wiwName', 'emdName']).size().reset_index(name=f'{FETCH_MODE}_station_count')
    counts.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"[+] Saved {len(counts)} unique dongs to {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_data()

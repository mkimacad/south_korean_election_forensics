import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time

# ==========================================
# CONFIGURATION
# ==========================================
SERVICE_KEY = 'YOUR_DECODED_SERVICE_KEY_HERE'
ELECTION_ID = '20220309'  # e.g., 20th Pres (20220309), 21st Gen (20200415), 22nd Gen (20240410)
FETCH_MODE  = 'normal'    # 'normal' for Election Day, 'pre' for Early Voting
OUTPUT_FILE = f"stations_{FETCH_MODE}_{ELECTION_ID}.csv"

# ==========================================
# CRAWLER LOGIC
# ==========================================
def get_provinces_for_election(elec_id):
    """
    Returns the correct official province names based on the election date.
    The API strictly requires the legal name at the time of the election.
    """
    # The 15 baseline provinces whose names haven't changed recently
    base_provinces = [
        '서울특별시', '부산광역시', '대구광역시', '인천광역시', '광주광역시', 
        '대전광역시', '울산광역시', '세종특별자치시', '경기도', '충청북도', 
        '충청남도', '전라남도', '경상북도', '경상남도', '제주특별자치도'
    ]
    
    # For the 22nd General Election (2024) and beyond
    if elec_id >= '20240410':
        base_provinces.extend(['강원특별자치도', '전북특별자치도'])
    # For earlier elections (19th/20th Presidential, 21st General, etc.)
    else:
        base_provinces.extend(['강원도', '전라북도'])
        
    return base_provinces

def fetch_data():
    base_url = 'http://apis.data.go.kr/9760000/PolplcInfoInqireService2/'
    endpoint = 'getPolplcOtlnmapTrnsportInfoInqire' if FETCH_MODE == 'normal' else 'getPrePolplcOtlnmapTrnsportInfoInqire'
    
    # Dynamically get the correct province names based on the election date
    provinces = get_provinces_for_election(ELECTION_ID)
    
    all_stations = []
    
    for sd in provinces:
        print(f"[*] Fetching {FETCH_MODE} stations for {sd}...")
        page = 1
        
        while True:
            params = {
                'serviceKey': SERVICE_KEY, 
                'pageNo': str(page), 
                'numOfRows': '100',  # Enforcing the 100-row limit to avoid API truncation
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
                
                # If the server returns fewer than 100 items, we've hit the last page
                if len(items) < 100:
                    break
                    
                page += 1
                time.sleep(0.1) # Safety delay to prevent IP blocking
                
            except Exception as e:
                print(f"  [!] Error on Page {page} ({sd}): {e}")
                break

    if not all_stations:
        print("\n[!] Failed to retrieve data. Please check ELECTION_ID or SERVICE_KEY.")
        return

    df = pd.DataFrame(all_stations)
    print(f"\n[!] Successfully collected {len(df)} raw stations nationally.")
    
    # Group by dong and calculate the count
    counts = df.groupby(['sdName', 'wiwName', 'emdName']).size().reset_index(name=f'{FETCH_MODE}_station_count')
    counts.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"[+] Saved {len(counts)} dong-level records to {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_data()

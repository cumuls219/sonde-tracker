# Sonde Tracker RAW v4 Log-P Wide + Highlight

UPP RAW 원시자료 기반 3D Sonde Tracker입니다.

## 주요 기능
- 원시자료 업로드 방식 유지
- 실제 고도 / Skew-T형 Log-P 연직축 선택
- Log-P 선택 시 P(hPa)만 사용하며 100hPa 이상 자료만 표시
- Log-P 모드 전용 wide canvas 레이아웃
- 동서남북 방위 표시
- RH(상대습도) 표시명 통일
- Asc(m/m)는 m/min으로 해석하고 Asc(m/s) 환산 제공
- 구름 가능층, 역전층, 하층제트 후보 3D 강조 옵션
- 하층제트 후보 자동 탐지 기준 조정 가능

## 실행
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## 하층제트 후보 탐지 기준
기본값은 3 km 이하, 풍속 20 kt 이상, 주변층 대비 5 kt 이상 약화입니다. 이는 경량 진단용 후보 탐지이며 공식 판정값은 아닙니다.

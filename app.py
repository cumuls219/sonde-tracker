import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from parser_raw import parse_upp_raw
from analysis_utils import (
    latlon_to_xy_km, add_wind_components, downsample_by_seconds, downsample_by_altitude,
    cloud_layers, inversion_layers, low_level_jet_layers, thermo_summary, layer_mean_table
)

st.set_page_config(page_title="Sonde Tracker RAW v4 Log-P Wide + Highlight", page_icon="🎈", layout="wide")

st.title("🎈 Sonde Tracker RAW v4 Log-P Wide + Highlight")
st.caption("UPP RAW 원시자료 업로드는 유지하고, Log-P 선택 시 지상~100hPa 범위만 표시하고, 확대 시 축·범례 잘림을 줄이고, 하층제트·역전층 강조 옵션을 추가한 버전입니다.")


def metric_fmt(v, unit="", digits=1):
    if v is None or pd.isna(v):
        return "-"
    return f"{v:,.{digits}f}{unit}"


def variable_label(col):
    """원시자료 컬럼명을 업무용 표시명으로 변환."""
    labels = {
        "T(C)": "T(C) 기온",
        "U(%)": "RH(상대습도)",
        "Wspd(knot)": "Wspd(knot) 풍속",
        "Asc(m/m)": "Asc(m/m) 상승률(분당 m)",
        "Asc(m/s)": "Asc(m/s) 상승속도",
        "P(hPa)": "P(hPa) 기압",
        "Dew(deg)": "Dew(deg) 노점",
    }
    return labels.get(col, col)


def get_display_df(df, mode):
    if mode == "전체":
        return df.copy()
    if mode == "2초 간격":
        return downsample_by_seconds(df, 2)
    if mode == "5초 간격":
        return downsample_by_seconds(df, 5)
    if mode == "10초 간격":
        return downsample_by_seconds(df, 10)
    if mode == "고도 50m 간격":
        return downsample_by_altitude(df, 50)
    if mode == "고도 100m 간격":
        return downsample_by_altitude(df, 100)
    return downsample_by_seconds(df, 5)



def prepare_vertical_axis(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """3D 표시용 z축을 실제 고도 또는 Log-P 좌표로 변환.

    - 실제 고도: Alt(m)/1000, 전체 표시
    - Skew-T형 Log-P: P(hPa)만 사용하며 지상~100hPa 범위(P>=100hPa)만 표시
    """
    out = df.copy()
    if mode == "Skew-T형 Log-P":
        if "P(hPa)" not in out.columns:
            raise ValueError("Log-P 축을 사용하려면 P(hPa) 컬럼이 필요합니다.")
        out["P(hPa)"] = pd.to_numeric(out["P(hPa)"], errors="coerce")
        # Log-P 모드에서는 그림 가림을 줄이고 기상학적으로 주로 쓰는 지상~100hPa 범위만 표시.
        # 압력 좌표에서 '100hPa까지'는 P >= 100hPa 조건에 해당함.
        out = out.loc[out["P(hPa)"].notna() & (out["P(hPa)"] >= 100.0)].copy()
        if len(out) < 2:
            raise ValueError("Log-P 축 표시를 위한 100hPa 이상 자료가 부족합니다.")
        p = out["P(hPa)"]
        p0 = float(p.iloc[0]) if p.notna().any() else 1000.0
        # Pressure-only Log-P coordinate. 배율은 눈금 표시용이며 실제 계산에는 영향 없음.
        out["z_plot"] = 7.0 * np.log(p0 / p.clip(lower=1.0))
    else:
        out["z_plot"] = out["Alt(m)"] / 1000.0
    return out


def vertical_axis_title(mode: str) -> str:
    if mode == "Skew-T형 Log-P":
        return "Log-P 연직축: P(hPa)"
    return "고도(km)"


def vertical_axis_ticks(df_display: pd.DataFrame, mode: str):
    """변환된 z_plot 좌표에 실제 고도/기압 라벨을 붙이기 위한 tick 설정."""
    if "z_plot" not in df_display.columns or len(df_display) == 0:
        return None, None

    if mode == "Skew-T형 Log-P":
        p = pd.to_numeric(df_display["P(hPa)"], errors="coerce").dropna()
        if len(p) < 2:
            return None, None
        p0 = float(p.iloc[0])
        p_min, p_max = float(p.min()), float(p.max())
        standard_ticks = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100]
        ticks = [pt for pt in standard_ticks if p_min <= pt <= p_max]
        # 지상기압이 1000hPa보다 높으면 첫 관측 기압도 보조 눈금으로 추가.
        if p_max > 1000 and all(abs(p_max - t) > 15 for t in ticks):
            ticks = [round(p_max)] + ticks
        if 100 not in ticks and p_min <= 110:
            ticks.append(100)
        ticks = sorted(set(ticks), reverse=True)
        zvals = [7.0 * np.log(p0 / max(pt, 1.0)) for pt in ticks]
        labels = [f"{pt:.0f} hPa" for pt in ticks]
        return zvals, labels

    max_alt = float(np.nanmax(df_display["Alt(m)"]))
    if not np.isfinite(max_alt):
        return None, None
    max_km = max_alt / 1000.0
    if max_km <= 2:
        alt_ticks_km = np.arange(0, math.ceil(max_km) + 0.5, 0.5)
    elif max_km <= 8:
        alt_ticks_km = np.arange(0, math.ceil(max_km) + 1, 1)
    else:
        alt_ticks_km = np.arange(0, math.ceil(max_km / 2) * 2 + 1, 2)
    return list(alt_ticks_km), [f"{v:g} km" for v in alt_ticks_km]



def padded_range(values, pad_ratio=0.08, min_pad=0.05):
    arr = pd.to_numeric(values, errors="coerce")
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    vmin = float(arr.min())
    vmax = float(arr.max())
    span = vmax - vmin
    if span <= 0:
        span = min_pad
    pad = max(span * pad_ratio, min_pad)
    return [vmin - pad, vmax + pad]

def make_hover(df, simple=True):
    if simple:
        return [
            f"시간: {r['Time(min:sec)']}<br>고도: {r['Alt(m)']:.1f} m<br>기온: {r['T(C)']:.1f}℃<br>RH(상대습도): {r['U(%)']:.1f}%<br>풍속: {r['Wspd(knot)']:.1f} kt<br>상승률: {r['Asc(m/m)']:.1f} m/min"
            for _, r in df.iterrows()
        ]
    return [
        f"시간: {r['Time(min:sec)']}<br>기압: {r['P(hPa)']:.1f} hPa<br>기온: {r['T(C)']:.1f}℃<br>노점: {r['Dew(deg)']:.1f}℃<br>RH(상대습도): {r['U(%)']:.1f}%<br>풍향/풍속: {r['Wdir(deg)']:.0f}° / {r['Wspd(knot)']:.1f} kt<br>위도: {r['Lat(deg)']:.5f}<br>경도: {r['Lon(deg)']:.5f}<br>고도: {r['Alt(m)']:.1f} m<br>상승률: {r['Asc(m/m)']:.1f} m/min = {r['Asc(m/s)']:.2f} m/s"
        for _, r in df.iterrows()
    ]



def add_direction_guide(fig, df_display):
    """3D 공간에서 동·서·남·북 방향 기준선을 표시."""
    if df_display is None or len(df_display) == 0:
        return fig
    max_extent = float(np.nanmax(np.sqrt(df_display["x_km"] ** 2 + df_display["y_km"] ** 2)))
    if not np.isfinite(max_extent) or max_extent <= 0:
        max_extent = 1.0
    guide_len = max(1.0, max_extent * 0.18)
    z0 = float(np.nanmin(df_display["z_plot"]))

    directions = [
        ("동(E)", guide_len, 0.0),
        ("서(W)", -guide_len, 0.0),
        ("북(N)", 0.0, guide_len),
        ("남(S)", 0.0, -guide_len),
    ]

    for label, x, y in directions:
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[z0, z0],
            mode="lines",
            line=dict(width=3, color="rgba(120,120,120,0.55)"),
            name=label,
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z0],
            mode="text",
            text=[label],
            textposition="middle center",
            textfont=dict(size=14, color="rgba(55,55,55,0.95)"),
            name=label,
            hoverinfo="skip",
            showlegend=False,
        ))
    return fig

def make_3d_fig(df_display, color_col, simple_hover=True, cloud=None, inversions=None, llj_layers=None, show_cloud=False, show_inversion=False, show_llj=False, show_direction=True, vertical_mode="실제 고도"):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df_display["x_km"], y=df_display["y_km"], z=df_display["z_plot"],
        mode="lines", line=dict(width=4, color="rgba(80,80,80,0.42)"),
        name="궤적선", hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter3d(
        x=df_display["x_km"], y=df_display["y_km"], z=df_display["z_plot"],
        mode="markers",
        marker=dict(size=4, color=df_display[color_col], colorscale="RdBu_r", showscale=True,
                    colorbar=dict(title=variable_label(color_col), len=0.62, y=0.58, x=0.98, thickness=16), opacity=0.88),
        text=make_hover(df_display, simple=simple_hover), hoverinfo="text", name=variable_label(color_col)
    ))
    # cloud possible points 강조: 3D 박스 대신 작은 마커로 경량 표시
    if show_cloud and cloud:
        masks = []
        for lyr in cloud:
            masks.append((df_display["Alt(m)"] >= lyr["base_m"]) & (df_display["Alt(m)"] <= lyr["top_m"]))
        if masks:
            mask = np.logical_or.reduce(masks)
            cdf = df_display.loc[mask]
            if len(cdf):
                fig.add_trace(go.Scatter3d(
                    x=cdf["x_km"], y=cdf["y_km"], z=cdf["z_plot"],
                    mode="markers", marker=dict(size=7, symbol="circle-open", color="rgba(165,165,165,0.72)", line=dict(color="rgba(120,120,120,0.55)", width=2)),
                    name="구름 가능층", hoverinfo="skip"
                ))

    # 역전층 강조: 선택 시 해당 고도 구간의 관측점을 보라색 open marker로 표시
    if show_inversion and inversions:
        masks = []
        for lyr in inversions:
            masks.append((df_display["Alt(m)"] >= lyr["base_m"]) & (df_display["Alt(m)"] <= lyr["top_m"]))
        if masks:
            mask = np.logical_or.reduce(masks)
            idf = df_display.loc[mask]
            if len(idf):
                fig.add_trace(go.Scatter3d(
                    x=idf["x_km"], y=idf["y_km"], z=idf["z_plot"],
                    mode="markers",
                    marker=dict(size=7, symbol="diamond-open", color="rgba(142,68,173,0.78)", line=dict(color="rgba(142,68,173,0.72)", width=2)),
                    name="역전층 강조", hoverinfo="skip"
                ))

    # 하층제트 강조: 선택 시 LLJ 후보 코어 주변을 주황색 open marker로 표시
    if show_llj and llj_layers:
        masks = []
        for lyr in llj_layers:
            masks.append((df_display["Alt(m)"] >= lyr["base_m"]) & (df_display["Alt(m)"] <= lyr["top_m"]))
        if masks:
            mask = np.logical_or.reduce(masks)
            jdf = df_display.loc[mask]
            if len(jdf):
                fig.add_trace(go.Scatter3d(
                    x=jdf["x_km"], y=jdf["y_km"], z=jdf["z_plot"],
                    mode="markers",
                    marker=dict(size=8, symbol="square-open", color="rgba(230,126,34,0.82)", line=dict(color="rgba(230,126,34,0.78)", width=2)),
                    name="하층제트 강조", hoverinfo="skip"
                ))
    # start/end/max height only
    points = [
        (df_display.iloc[0], "START", "green", "circle"),
        (df_display.iloc[-1], "END", "black", "x"),
        (df_display.loc[df_display["Alt(m)"].idxmax()], "MAX", "purple", "diamond"),
    ]
    for row, label, color, symbol in points:
        fig.add_trace(go.Scatter3d(
            x=[row["x_km"]], y=[row["y_km"]], z=[row["z_plot"]],
            mode="markers+text", marker=dict(size=8, color=color, symbol=symbol),
            text=[label], textposition="top center", name=label
        ))
    if show_direction:
        add_direction_guide(fig, df_display)

    tick_vals, tick_text = vertical_axis_ticks(df_display, vertical_mode)
    zaxis_cfg = dict(title=vertical_axis_title(vertical_mode))
    if tick_vals is not None and tick_text is not None:
        zaxis_cfg.update(dict(tickmode="array", tickvals=tick_vals, ticktext=tick_text))

    # 확대/회전 시 하단 눈금과 범례가 잘리지 않도록 장면 영역과 여백을 분리.
    x_range = padded_range(df_display["x_km"], pad_ratio=0.10, min_pad=0.5)
    y_range = padded_range(df_display["y_km"], pad_ratio=0.10, min_pad=0.5)
    z_range = padded_range(df_display["z_plot"], pad_ratio=0.08, min_pad=0.25)
    zaxis_cfg.update(dict(range=z_range))

    is_logp = vertical_mode == "Skew-T형 Log-P"

    scene_cfg = dict(
        xaxis=dict(title="동서 이동거리(km, +동쪽)", range=x_range),
        yaxis=dict(title="남북 이동거리(km, +북쪽)", range=y_range),
        zaxis=zaxis_cfg,
        # 3D 장면 자체는 크게 쓰되, Log-P 모드에서는 아래쪽 압력축 라벨이 잘리지 않도록
        # 하단 여백을 더 확보한다.
        domain=dict(x=[0.03, 0.86] if is_logp else [0.02, 0.90],
                    y=[0.11, 0.94] if is_logp else [0.10, 0.95]),
    )
    if is_logp:
        scene_cfg.update(dict(
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=2.25),
            camera=dict(eye=dict(x=1.65, y=1.85, z=1.05), center=dict(x=0, y=0, z=-0.10)),
        ))
    else:
        scene_cfg.update(dict(aspectmode="data"))

    if is_logp:
        fig_height = 1120
        fig_margin = dict(l=30, r=170, t=85, b=185)
        legend_cfg = dict(
            orientation="v",
            yanchor="top", y=0.93,
            xanchor="left", x=0.875,
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(210,210,210,0.75)", borderwidth=1,
            font=dict(size=11),
        )
    else:
        fig_height = 780
        fig_margin = dict(l=20, r=60, t=70, b=115)
        legend_cfg = dict(
            orientation="h",
            yanchor="bottom", y=0.02,
            xanchor="left", x=0.02,
            bgcolor="rgba(255,255,255,0.75)",
            bordercolor="rgba(210,210,210,0.6)", borderwidth=1,
        )

    fig.update_layout(
        title=dict(text="GPS 기반 3D Sonde Tracker", x=0.5, y=0.985),
        scene=scene_cfg,
        height=fig_height,
        margin=fig_margin,
        legend=legend_cfg
    )
    return fig


def make_profile_fig(df, clouds=None, inversions=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["T(C)"], y=df["Alt(m)"]/1000, mode="lines", name="기온"))
    fig.add_trace(go.Scatter(x=df["Dew(deg)"], y=df["Alt(m)"]/1000, mode="lines", name="노점"))
    if "Parcel_T(C)" in df.columns:
        fig.add_trace(go.Scatter(x=df["Parcel_T(C)"], y=df["Alt(m)"]/1000, mode="lines", name="상승기온(근사)"))
    if clouds:
        for lyr in clouds:
            fig.add_hrect(y0=lyr["base_km"], y1=lyr["top_km"], fillcolor="rgba(170,170,170,0.22)", opacity=0.22, line_width=0, annotation_text="구름 가능층")
    if inversions:
        for lyr in inversions:
            fig.add_hrect(y0=lyr["base_km"], y1=lyr["top_km"], opacity=0.12, line_width=0, annotation_text="inv")
    fig.update_layout(title="기온·노점 프로파일", xaxis_title="℃", yaxis_title="고도(km)", height=620, margin=dict(l=10,r=10,t=50,b=10))
    return fig


def make_rh_fig(df, clouds=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["U(%)"], y=df["Alt(m)"]/1000, mode="lines", name="RH(상대습도)"))
    if clouds:
        for lyr in clouds:
            fig.add_hrect(y0=lyr["base_km"], y1=lyr["top_km"], fillcolor="rgba(170,170,170,0.22)", opacity=0.22, line_width=0)
    fig.update_layout(title="상대습도 프로파일", xaxis_title="RH(상대습도, %)", yaxis_title="고도(km)", height=620, margin=dict(l=10,r=10,t=50,b=10))
    return fig


def make_wind_profile(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Wspd(knot)"], y=df["Alt(m)"]/1000, mode="lines", name="풍속"))
    fig.update_layout(title="고도별 풍속", xaxis_title="kt", yaxis_title="고도(km)", height=560, margin=dict(l=10,r=10,t=50,b=10))
    return fig


def make_asc_profile(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Asc(m/m)"], y=df["Alt(m)"]/1000, mode="lines", name="Asc(m/m) 상승률"))
    fig.update_layout(title="고도별 상승률", xaxis_title="Asc(m/m) = m/min, 60으로 나누면 m/s", yaxis_title="고도(km)", height=560, margin=dict(l=10,r=10,t=50,b=10))
    return fig


def make_hodograph(df, display_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=display_df["u_kt"], y=display_df["v_kt"], mode="lines+markers",
        marker=dict(size=5, color=display_df["Alt(m)"], colorscale="Viridis", colorbar=dict(title="Alt(m)")),
        text=[f"{r['Alt(m)']:.0f} m<br>{r['Wspd(knot)']:.1f} kt / {r['Wdir(deg)']:.0f}°" for _, r in display_df.iterrows()],
        hoverinfo="text", name="hodograph"
    ))
    max_abs = np.nanmax(np.abs(pd.concat([df["u_kt"], df["v_kt"]])))
    max_abs = max(10, math.ceil(max_abs/10)*10)
    fig.update_layout(title="호도그래프", xaxis_title="u(kt)", yaxis_title="v(kt)", height=650,
                      xaxis=dict(range=[-max_abs,max_abs], zeroline=True), yaxis=dict(range=[-max_abs,max_abs], zeroline=True, scaleanchor="x", scaleratio=1),
                      margin=dict(l=10,r=10,t=50,b=10))
    return fig

with st.sidebar:
    st.header("1. 원시자료 업로드")
    uploaded = st.file_uploader("UPP RAW TXT 파일", type=["txt", "dat", "csv", "log"])
    st.caption("업로드 방식은 v1/v2와 동일합니다. 내부에서만 표시 자료를 줄입니다.")
    st.divider()
    st.header("2. 경량 표시 설정")
    display_mode = st.selectbox("3D 표시 간격", ["5초 간격", "10초 간격", "2초 간격", "고도 50m 간격", "고도 100m 간격", "전체"], index=0)
    vertical_mode = st.selectbox("3D 연직축 표현", ["실제 고도", "Skew-T형 Log-P"], index=0, help="기본은 실제 고도입니다. Skew-T형 Log-P는 P(hPa)만 사용하며 지상~100hPa 범위만 표시하고, 이때만 연직축을 길게 표현합니다.")
    color_col = st.selectbox("3D 색상 변수", ["T(C)", "U(%)", "Wspd(knot)", "Asc(m/m)"], index=0, format_func=variable_label)
    simple_hover = st.checkbox("간단 hover 사용", value=True)
    show_direction_guide = st.checkbox("동·서·남·북 방위 표시", value=True)
    st.divider()
    st.header("3. 구름 가능층")
    show_cloud_3d = st.checkbox("3D에서 구름 가능층 강조", value=False)
    show_inversion_3d = st.checkbox("3D에서 역전층 강조", value=False)
    show_llj_3d = st.checkbox("3D에서 하층제트 강조", value=False)
    rh_th = st.slider("RH(상대습도) 기준(%)", 70, 100, 85, 1)
    spread_th = st.slider("T-Td 기준(℃)", 0.5, 5.0, 2.0, 0.5)
    min_cloud_thick = st.slider("최소 층 두께(m)", 20, 300, 100, 10)
    st.divider()
    st.header("4. 하층제트 탐지")
    llj_max_alt = st.slider("LLJ 탐지 상한고도(m)", 1000, 5000, 3000, 500)
    llj_min_speed = st.slider("LLJ 최소 풍속(kt)", 10, 50, 20, 1)
    llj_drop = st.slider("LLJ 풍속 감소 기준(kt)", 2, 20, 5, 1)

if uploaded is None:
    st.info("왼쪽에서 UPP RAW 원시자료 TXT 파일을 업로드하세요.")
    st.markdown("""
    **v4 Log-P Axis 기본값**
    - 3D는 5초 간격 자료만 표시
    - 원본 전체 자료는 계산과 CSV 저장에 사용
    - 연직축은 실제 고도 또는 Skew-T형 Log-P 중 선택
    - Log-P 선택 시 P(hPa)만 이용하고 지상~100hPa 범위만 표시
    - 구름층/역전층/하층제트 3D 강조는 기본 OFF, 필요 시 강조 마커만 표시
    - 열역학 분석은 해당 탭에서 버튼을 눌렀을 때 계산
    """)
    st.stop()

try:
    raw_df, info = parse_upp_raw(uploaded)
    raw_df = latlon_to_xy_km(raw_df)
    raw_df = add_wind_components(raw_df)
    if "Asc(m/m)" in raw_df.columns:
        raw_df["Asc(m/s)"] = raw_df["Asc(m/m)"] / 60.0
except Exception as e:
    st.error("원시자료 파싱 중 오류가 발생했습니다.")
    st.exception(e)
    st.stop()

display_df = get_display_df(raw_df, display_mode)
display_df = prepare_vertical_axis(display_df, vertical_mode)
clouds = cloud_layers(raw_df, rh_threshold=rh_th, spread_threshold=spread_th, min_thickness_m=min_cloud_thick)
inversions = inversion_layers(raw_df)
llj_layers = low_level_jet_layers(raw_df, max_alt_m=llj_max_alt, min_speed_kt=llj_min_speed, drop_threshold_kt=llj_drop)

# Header metrics
c1, c2, c3, c4, c5, c6 = st.columns(6)
with c1: st.metric("원본 행 수", f"{len(raw_df):,}")
with c2: st.metric("3D 표시 행 수", f"{len(display_df):,}")
with c3: st.metric("최대 고도", metric_fmt(raw_df["Alt(m)"].max(), " m", 1))
with c4: st.metric("최저 기온", metric_fmt(raw_df["T(C)"].min(), "℃", 1))
with c5: st.metric("구름 가능층", f"{len(clouds)}개")
with c6: st.metric("LLJ 후보", f"{len(llj_layers)}개")

if info:
    st.caption(f"Station {info.get('station_no')} | Lat {info.get('latitude')} | Lon {info.get('longitude')} | Alt {info.get('altitude_m')} m | Probe {info.get('probe_no')}")
st.caption("※ 원시자료의 Asc(m/m)는 meter/minute, 즉 분당 상승률(m/min)로 표시합니다. m/s 환산값은 Asc(m/m) ÷ 60입니다.")

tab1, tab2, tab3, tab4 = st.tabs(["3D Tracker", "열역학", "바람·호도그래프", "자료/다운로드"])

with tab1:
    st.subheader("3D Tracker")
    st.caption("계산은 원본 전체 자료를 쓰고, 3D 표시만 선택 간격으로 줄입니다. Log-P 선택 시 P(hPa)만 사용하며 지상~100hPa 범위(P≥100hPa)만 표시합니다.")
    fig = make_3d_fig(display_df, color_col=color_col, simple_hover=simple_hover, cloud=clouds, inversions=inversions, llj_layers=llj_layers, show_cloud=show_cloud_3d, show_inversion=show_inversion_3d, show_llj=show_llj_3d, show_direction=show_direction_guide, vertical_mode=vertical_mode)
    st.plotly_chart(fig, use_container_width=True)
    html = fig.to_html(include_plotlyjs="cdn", full_html=True).encode("utf-8")
    st.download_button("표시 중인 3D HTML 다운로드", html, "sonde_tracker_3d_light.html", "text/html")

with tab2:
    st.subheader("열역학·기온 관련 진단")
    st.caption("빠른 로딩을 위해 CAPE/CIN 계산은 버튼을 눌렀을 때만 실행합니다.")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(make_profile_fig(raw_df, clouds=clouds, inversions=inversions), use_container_width=True)
    with right:
        st.plotly_chart(make_rh_fig(raw_df, clouds=clouds), use_container_width=True)
    st.markdown("#### 자동 탐지 결과")
    a, b = st.columns(2)
    with a:
        st.write("구름 가능층")
        st.dataframe(pd.DataFrame(clouds), use_container_width=True, hide_index=True)
    with b:
        st.write("역전층")
        st.dataframe(pd.DataFrame(inversions), use_container_width=True, hide_index=True)
    if st.button("CAPE/CIN/LCL/CCL/대류온도 계산 실행", type="primary"):
        summary, parcel_profile = thermo_summary(raw_df)
        cols = st.columns(4)
        for i, (k, v) in enumerate(summary.items()):
            with cols[i % 4]:
                unit = "" if "J/kg" in k else (" m" if "(m)" in k else "℃")
                digits = 0 if "J/kg" in k or "(m)" in k else 1
                st.metric(k, metric_fmt(v, unit, digits))
        st.plotly_chart(make_profile_fig(parcel_profile, clouds=clouds, inversions=inversions), use_container_width=True)
        st.info("CAPE/CIN/대류온도는 경량 자체식 기반 근사값입니다. 공식 현업 산출용 정밀값은 MetPy 등으로 후속 고도화가 필요합니다.")

with tab3:
    st.subheader("바람·상승률·호도그래프")
    st.caption("Asc(m/m)는 원시자료 표기 그대로 유지하되, 단위 의미는 m/min(분당 m)입니다. 예: 300 m/min ≈ 5.0 m/s")
    if llj_layers:
        st.markdown("#### 하층제트 후보")
        st.dataframe(pd.DataFrame(llj_layers), use_container_width=True, hide_index=True)
    else:
        st.info("현재 기준에서 하층제트 후보가 탐지되지 않았습니다. 필요하면 사이드바의 LLJ 기준값을 조정하세요.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_wind_profile(raw_df), use_container_width=True)
    with c2:
        st.plotly_chart(make_asc_profile(raw_df), use_container_width=True)
    st.plotly_chart(make_hodograph(raw_df, display_df), use_container_width=True)
    st.markdown("#### 1 km 층별 평균")
    st.dataframe(layer_mean_table(raw_df), use_container_width=True, hide_index=True)

with tab4:
    st.subheader("자료 확인 및 다운로드")
    st.caption("화면에는 일부만 보여주고, 전체 자료는 다운로드로 제공합니다.")
    st.markdown("#### 원본 미리보기 50행")
    st.dataframe(raw_df.head(50), use_container_width=True)
    st.markdown("#### 표시용 자료 미리보기 50행")
    st.dataframe(display_df.head(50), use_container_width=True)
    st.download_button("원본 변환 CSV 다운로드", raw_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "sonde_raw_converted_full.csv", "text/csv")
    st.download_button("표시용 경량 CSV 다운로드", display_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"), "sonde_display_light.csv", "text/csv")

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# 페이지 설정 (브라우저 탭 제목 및 레이아웃)
st.set_page_config(page_title="Bitcoin Analysis Dashboard", layout="wide")

@st.cache_data
def load_data(file_path):
    """
    동일 폴더의 CSV 파일을 읽어와 전처리하는 함수
    """
    if not os.path.exists(file_path):
        return None
    
    # 데이터 로드 (세미콜론 구분자 처리)
    df = pd.read_csv(file_path, sep=';')
    
    # 시간 관련 컬럼을 datetime 객체로 변환 (ISO8601 형식 대응)
    df['timeOpen'] = pd.to_datetime(df['timeOpen'])
    
    # 분석에 필요한 수치형 데이터 확인
    cols_to_fix = ['open', 'high', 'low', 'close', 'volume']
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 날짜순으로 정렬
    df = df.sort_values('timeOpen')
    return df

def predict_next_day(df):
    """
    선형 회귀 모델을 사용하여 내일 가격을 예측하는 함수
    """
    # 학습을 위해 날짜를 숫자로 변환 (Ordinal)
    df_copy = df.dropna(subset=['close']).copy()
    df_copy['date_ordinal'] = df_copy['timeOpen'].map(datetime.toordinal)
    
    X = df_copy[['date_ordinal']].values
    y = df_copy['close'].values
    
    # 모델 학습
    model = LinearRegression()
    model.fit(X, y)
    
    # 내일 날짜 계산 및 예측
    next_day = df_copy['timeOpen'].max() + timedelta(days=1)
    next_day_ordinal = np.array([[next_day.toordinal()]])
    prediction = model.predict(next_day_ordinal)[0]
    
    return next_day, prediction, model

def main():
    st.title("🪙 비트코인 실시간 데이터 및 AI 예측")
    st.info("현재 폴더의 `coin.csv` 데이터를 분석하고 내일의 가격을 예측합니다.")

    # 1. 데이터 불러오기
    csv_filename = 'coin.csv'
    df = load_data(csv_filename)

    if df is None:
        st.error(f"파일을 찾을 수 없습니다: `{csv_filename}` 파일이 파이썬 파일과 같은 폴더에 있는지 확인해주세요.")
        return

    # 2. 사이드바 필터 구성
    st.sidebar.header("📊 분석 및 예측 설정")
    
    # 날짜 범위 선택
    min_date = df['timeOpen'].min().date()
    max_date = df['timeOpen'].max().date()
    
    date_range = st.sidebar.date_input(
        "분석 기간",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # 3. 데이터 필터링 적용
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['timeOpen'].dt.date >= start_date) & (df['timeOpen'].dt.date <= end_date)
        filtered_df = df.loc[mask].copy()
    else:
        filtered_df = df.copy()

    if filtered_df.empty:
        st.warning("선택한 기간에 해당하는 데이터가 없습니다.")
        return

    # 4. 내일 가격 예측 실행
    next_date, pred_price, model = predict_next_day(filtered_df)
    current_price = filtered_df['close'].iloc[-1]
    is_up = pred_price > current_price

    # 5. 주요 지표 (KPI) 섹션
    st.subheader("🚀 AI 시장 예측")
    p1, p2, p3 = st.columns(3)
    
    p1.metric("현재 종가", f"₩{current_price:,.0f}")
    p2.metric(f"내일 예측가 ({next_date.strftime('%Y-%m-%d')})", f"₩{pred_price:,.0f}", 
              f"{((pred_price-current_price)/current_price)*100:+.2f}%")
    
    if is_up:
        p3.success("📈 AI 추천: 가격 상승이 예상됩니다. (매수 고려)")
    else:
        p3.error("📉 AI 추천: 가격 하락이 예상됩니다. (관망 권장)")

    st.divider()

    # 6. 메인 차트 (캔들스틱 + 예측선)
    st.subheader("📈 가격 변동 및 예측 트렌드")
    
    # 이동평균선(MA) 계산
    filtered_df['MA20'] = filtered_df['close'].rolling(window=20).mean()
    
    # 회귀선 데이터 생성
    df_ordinal = filtered_df['timeOpen'].map(datetime.toordinal).values.reshape(-1, 1)
    filtered_df['TrendLine'] = model.predict(df_ordinal)

    fig = go.Figure()

    # 캔들스틱 추가
    fig.add_trace(go.Candlestick(
        x=filtered_df['timeOpen'],
        open=filtered_df['open'],
        high=filtered_df['high'],
        low=filtered_df['low'],
        close=filtered_df['close'],
        name='BTC'
    ))

    # 추세선(회귀선) 추가
    fig.add_trace(go.Scatter(
        x=filtered_df['timeOpen'], 
        y=filtered_df['TrendLine'], 
        name='AI 추세선', 
        line=dict(color='red', width=2, dash='dash')
    ))

    # 이동평균선 추가
    fig.add_trace(go.Scatter(x=filtered_df['timeOpen'], y=filtered_df['MA20'], name='MA20', line=dict(color='rgba(255, 165, 0, 0.8)')))

    fig.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 7. 거래량 및 데이터 테이블
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📊 거래량 분석")
        vol_fig = go.Figure(data=[
            go.Bar(x=filtered_df['timeOpen'], y=filtered_df['volume'], marker_color='lightslategrey')
        ])
        vol_fig.update_layout(height=300, margin=dict(t=10), template="plotly_white")
        st.plotly_chart(vol_fig, use_container_width=True)

    with col_right:
        st.subheader("🔍 최근 데이터")
        recent_data = filtered_df[['timeOpen', 'close', 'volume']].sort_values('timeOpen', ascending=False).head(10)
        st.table(recent_data.style.format({'close': '{:,.0f}', 'volume': '{:,.0f}'}))

if __name__ == "__main__":
    main()

import streamlit as st
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from datetime import datetime
import FinanceDataReader as fdr
import pandas as pd
from enum import Enum

class Ticker(Enum):
    코스닥인버스 = '251340'
    이차전지 = '305540'
    자동차 = '091180'
    중국소비 = '150460'

class History:
    def __init__(self, year=1) -> None:
        self._year = year
        self.data = {t: self._getDataFromTicker(t) for t in Ticker}
        
    def _getDataFromTicker(self, ticker: Ticker):
        df = self._getClosePrice(ticker.value, self._year)
        df.index = df.index.date
        return df 
    
    @staticmethod
    def _getClosePrice(ticker: str, year: int=1):
        return fdr.DataReader(ticker, datetime.now().year - year)['Close']

class Predictor:
    def __init__(self, data: History, period:int=20, shift=0) -> None:
        self._data = data
        self.period = period
        self._shift = shift
        self.pred = {t: self._getEarningRate(t) for t in Ticker}

    def _getEarningRate(self, ticker: Ticker):
        df = self._getDFFromTicker(ticker)
        return (df.iloc[-1-self._shift]
                / df.iloc[-self.period-1-self._shift]) - 1

    def _getDFFromTicker(self, ticker: Ticker) -> pd.DataFrame:
        return self._data.data[ticker]

def get_momentum(cd: History, shift=0,
                #  span=(3, 5, 8, 13, 21),
                #  span=(5, 8, 13, 21, 34),
                 span=(8, 13, 21, 34, 55),
                 weight=True):
                #  weight=True):
    pdt_period = []
    for s in span:
        pdt = Predictor(cd, s, shift)
        period = pdt.period
        w = (pdt.period if weight else 1)
        d = { "period" : period}
        for t in Ticker:
            d[t.name] = pdt.pred[t] / w * 100
        pdt_period.append(d)
    pdt_df = pd.DataFrame(pdt_period).set_index('period')
    return pdt_df.sum().sort_values(ascending=False) / 5

def visualize(cd, period, weight, span):
    # 각 shift 값에 대해 모멘텀을 계산하고, DataFrame으로 저장합니다.
    df = pd.concat([get_momentum(cd, i, weight=weight)
                    for i in range(period, -1, -1)], axis=1
                   ).T.ewm(span).mean()

    # 테이블
    st.title("Momentum Score Dashboard")
    table = df.iloc[-1].sort_values(ascending=False)
    table.name = 'Score'

    # 그래프
    # df.T.plot(style='--x')    
    st.dataframe(table, use_container_width=True)
    fig = go.Figure()
    for t in Ticker:
        options = {
            'x': df.index, 'y': df[t.name],
            'mode': 'lines+markers', 'name': f"{t.name}({t.value})"
        }
        fig.add_trace(go.Scatter(**options))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        width=500,
        height=250,
        # legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        legend=dict(orientation='h', yanchor='top', y=-0.1, xanchor='center', x=0.5)
    )
    # fig.show()
    # col1, col2 = st.columns([2, 3])
    st.plotly_chart(fig, True, theme=None)
    # return df.iloc[-1]

    # 이중축
    # st.write(table.index[0], table.index[1])
    first, second = table.index[:2]
    d1 = cd.data[getattr(Ticker, first)].iloc[-20:]
    d2 = cd.data[getattr(Ticker, second)].iloc[-20:]

    # Plotly 서브플롯 생성
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # 데이터를 서브플롯에 추가
    fig.add_trace(go.Scatter(y=d1, mode='lines', name=first), secondary_y=False)
    fig.add_trace(go.Scatter(y=d2, mode='lines', name=second), secondary_y=True)

    # 축 레이블 설정
    fig.update_yaxes(title_text=f'{first} 가격', secondary_y=False)
    fig.update_yaxes(title_text=f'{second} 가격', secondary_y=True)

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        width=500,
        height=250,
        # legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        legend=dict(orientation='h', yanchor='top', y=1.02, xanchor='center', x=0.5)
    )
    
    # Stremlit에 Plotly 그래프 표시
    st.plotly_chart(fig, True)

data = History()
PERIOD = 20
WEIGHT = True
SPAN = 3

visualize(data, PERIOD, WEIGHT, SPAN)

#%%
import MetaTrader5 as mt5
import pandas as pd
import time
import pytz
import plotly.graph_objects as go
import cufflinks as cf
import numpy as np
import seaborn as sns
import pyarrow.parquet as pq
from datetime import datetime
from scipy.signal import argrelextrema
from plotly.subplots import make_subplots


cf.set_config_file(offline = True)

if not mt5.initialize():
    print("initialize() failed, error code =",mt5.last_error())
    quit()

# %%
symbol = 'WDO@N'

'''dateBegin =  datetime(2022,4,18,0,0)
dateEnd = datetime(2022,4,19,0)

rates_df = pd.DataFrame(mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, dateBegin, dateEnd))
rates_df_2 = rates_df
rates_df['time'] = pd.to_datetime(rates_df['time'], unit='s')

rates_df.set_index('time', inplace=True)'''

rates_df = pq.ParquetFile("output.parquet").read().to_pandas()
rates_df.set_index('time', inplace=True)
rates_df.index = pd.to_datetime(rates_df.index, format="%Y-%m-%d %H:%M:%S")


# %%
#Armazenando high locais
df_ = rates_df.copy()
df_["i"] = np.arange(len(df_))

local_max_index = np.array(argrelextrema(rates_df.high.values, np.greater, order=15, mode='wrap')[0])

local_max=[]

for loc in local_max_index:
  local_max.append(df_.high[loc])

local_max=np.array(local_max)
local_max

df_["local_max"] = 0
df_.loc[df_["i"].isin(local_max_index), "local_max"] = 1

# %%
def round_limit_order_price(limitOrderPrice):
  decimal_value = limitOrderPrice % 1

  if decimal_value > 0.25 and decimal_value <= 0.5:
    limitOrderPrice = float(round(limitOrderPrice) + 0.5)
  elif decimal_value > 0.5 and decimal_value <= 0.75:
    limitOrderPrice = float(round(limitOrderPrice) - 0.5)
  else:
    limitOrderPrice = float(round(limitOrderPrice))

  return limitOrderPrice

#Variáveis de configuração
value_per_pip = 5
order_tax = 0
trade_volume = 1

extra_range_entry = 0.5
n_bars_validation = 5
entry_threshold_attemp = 30
operation_duration = 5
order_to_be_filled_threshold = 30

trade_hour_threshold = 17
trade_minute_treshold = 20
#

df_["local_entry"] = 0
df_["local_position_close"] = 0
df_["entry_value"] = 0
total_entries = 0
profit = []

ignored_pullbacks = 0
ignored_limit_orders = 0
for i in range(1,len(local_max_index)):

  #Permitindo apenas canais diários
  if (df_[local_max_index[i-1]:(local_max_index[i-1]+1)].index.day[0] != df_[local_max_index[i]:(local_max_index[i]+1)].index.day[0]):
    continue

  firstPoint = df_[local_max_index[i-1]:(local_max_index[i-1]+1)].high.values[0]
  secondPoint = df_[local_max_index[i]:(local_max_index[i]+1)].high.values[0]

  barDistance_pips = firstPoint - secondPoint
  barDistance_range = local_max_index[i] - local_max_index[i-1]

  j=0
  calculatingPullbackEntry = False
  ordersTotal = False
  positionsTotal = False
  limitOrderPrice = 0
  orderRunner=0

  #print("i = ", i)
  #corrigir tamanho max desse for para menos de 540 min
  for close in df_[(local_max_index[i]+1):len(df_)].close.values:
    j += 1
    #Não pode ultrapassar horário limite
    if (df_[(local_max_index[i]+j):(local_max_index[i]+1+j)].index.hour[0]) >= trade_hour_threshold and (df_[(local_max_index[i]+j):(local_max_index[i]+1+j)].index.minute[0]) > trade_minute_treshold:
      #print('Time to trade is over', df_[(local_max_index[i]+j):(local_max_index[i]+1+j)].index.time[0])
      #print('OrdersTotal:', ordersTotal, 'PositionsTotal:', positionsTotal)
      break

    lineVariation = (barDistance_pips*(barDistance_range+j))/barDistance_range
    breakOutLine = firstPoint - lineVariation

    #Verificando breakout
    if(close > breakOutLine and calculatingPullbackEntry == False):
      #--------------
      #print("Break out price:", close, "at", df_[(local_max_index[i]+j):(local_max_index[i]+1+j)].index.time[0])
      calculatingPullbackEntry = True
      j += (n_bars_validation-2)

    #Atualizando limit order (pullback validado)
    elif(calculatingPullbackEntry == True):
      if ordersTotal == True:
        orderRunner +=1
        limitOrderPrice = breakOutLine + extra_range_entry
        limitOrderPrice = round_limit_order_price(limitOrderPrice)

        #Se ativar ordem, cai aqui
        if df_[(local_max_index[i]+j):(local_max_index[i]+j+1)].low.values[0] <= limitOrderPrice:
          #print("Position created at", limitOrderPrice, "at", df_[(local_max_index[i]+j):(local_max_index[i]+1+j)].index.time[0], "limitOrder:", limitOrderPrice)
          positionsTotal = True  #plotar entrada no grafico
          total_entries += 1
          df_['local_entry'][(local_max_index[i]+j):(local_max_index[i]+j+1)] = 1
          df_['entry_value'][(local_max_index[i]+j):(local_max_index[i]+j+1)] = limitOrderPrice
          j=j+operation_duration
          df_['local_position_close'][(local_max_index[i]+j):(local_max_index[i]+j+1)] = 1

          #calcular distância do profit
          profit.append(((df_[(local_max_index[i]+j):(local_max_index[i]+j+1)].close.values - limitOrderPrice)*value_per_pip*trade_volume)+order_tax)
          #
          #print("Exit point:", df_[(local_max_index[i]+j):(local_max_index[i]+j+1)].close.values[0], df_[(local_max_index[i]+j):(local_max_index[i]+1+j)].index.time[0])
          break

        #Ordem limite precisa ser acionada em menos de x minutos
        if (orderRunner > order_to_be_filled_threshold):
          ignored_limit_orders += 1
          #print("Order couldn't be filled.", df_[(local_max_index[i]+j):(local_max_index[i]+1+j)].index.time[0])
          break

      #Validando pullback (breakout validado)
      elif (ordersTotal == False):

        if(df_[(local_max_index[i]+j):(local_max_index[i]+j+1)].low.values[0] > breakOutLine):
          #Abrindo ordem limite
          limitOrderPrice = breakOutLine + extra_range_entry
          limitOrderPrice = round_limit_order_price(limitOrderPrice)
          ordersTotal = True 
          #print("Opening first limit order at",limitOrderPrice, "limitOrder:", limitOrderPrice, "at", df_[(local_max_index[i]+j):(local_max_index[i]+1+j)].index.time[0])
        else:
          #Ignorando pullback
          ignored_pullbacks += 1
          #print("Pullback ignored")
          break


#----------Up -> inside for
profit = pd.DataFrame(profit)
print(sns.distplot(profit, bins=15))
print("Total entries:", total_entries,
      "\nIgnored pullbacks:", ignored_pullbacks,
      "\nIgnored limit orders:", ignored_limit_orders,
      "\nMean:", profit[0].mean(),
      "\nStd:", profit[0].std(),)



# %%
#Plotando gráfico: exibindo max locais, entradas e saídas

dateBegin =  datetime(2022,3,31,0,0)
dateEnd = datetime(2022,4,1,0)

df_2 = df_[(df_.index >= dateBegin) & (df_.index <= dateEnd)].copy()

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
                    vertical_spacing=0.03, subplot_titles=('OHLC', 'Volume'), row_width=[0.2, 0.7])
                    
fig.add_trace(go.Candlestick(x=df_2.index, open=df_2['open'], high=df_2['high'], low=df_2['low'], close=df_2['close']), row=1, col=1)
fig.add_trace(go.Scatter(x=df_2[df_2["local_max"] == 1].index, y=df_2[df_2["local_max"] == 1]["high"], mode="markers", marker_color="cyan", marker_symbol="x", marker_size=15, opacity=0.5), row=1, col=1)
fig.add_trace(go.Scatter(x=df_2[df_2["local_entry"] == 1].index, y=df_2[df_2["local_entry"] == 1]["entry_value"], mode="markers", marker_color="cyan", marker_symbol="circle", marker_size=15, opacity=0.5), row=1, col=1)
fig.add_trace(go.Scatter(x=df_2[df_2["local_position_close"] == 1].index, y=df_2[df_2["local_position_close"] == 1]["close"], mode="markers", marker_color="cyan", marker_symbol="square", marker_size=15, opacity=0.5), row=1, col=1)

fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=700)

fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[18, 9], pattern="hour")])

fig.show()
# %%


# %%

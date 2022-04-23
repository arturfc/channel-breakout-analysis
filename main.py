#%%
import MetaTrader5 as mt5
import pandas as pd
import time
import plotly.graph_objects as go
import cufflinks as cf
import numpy as np
import seaborn as sns
import pyarrow.parquet as pq
from datetime import datetime
from scipy.signal import argrelextrema
from plotly.subplots import make_subplots
import ta

df = pq.ParquetFile("output.parquet").read().to_pandas()
df.set_index('time', inplace=True)
df.index = pd.to_datetime(df.index, format="%Y-%m-%d %H:%M:%S")

#adicionando média móvel simples de 200 períodos ou RSI
#df['SMA_200'] = df.close.rolling(200).mean()
df['rsi'] = ta.momentum.rsi(df.close, 7)

#Note que o mercado passa por duas grandes fases de tendência
df[["close"]].plot()
# %%
#Armazenando high locais
df["i"] = np.arange(len(df))

local_max_index = np.array(argrelextrema(df.high.values, np.greater, order=15, mode='wrap')[0])

local_max=[]
for loc in local_max_index:
  local_max.append(df.high[loc])

local_max=np.array(local_max)
local_max

df["local_max"] = 0
df.loc[df["i"].isin(local_max_index), "local_max"] = 1

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

#dados do ativo
value_per_pip = 10  #valor em R$ por variação de 1 ponto
order_tax = 1.33       #custo em R$ por ordem enviada

#dentro da operação
trade_volume = 1          #número de lotes/contratos
stop_gain_distance = 14    #distância do stop gain
stop_loss_distance = 6    #distância do stop loss
operation_time = 30       #duração máxima de uma operação

#validação do pullback
n_bars_validation = 10  #numero de barras para validar um pullback
extra_range_entry = 0   #range para entrar adiantado na operação
operation_duration = 15 #deprecated

order_to_be_filled_threshold = 30 #tempo de limite para consumir a ordem existente

trade_hour_threshold = 17   #horário limite de trade 
trade_minute_treshold = 0  #minuto limite de trade
#

df["local_entry"] = 0
df["entry_value"] = 0
df["exit"] = 0
df["exit_value"] = 0

ignored_pullbacks = 0
ignored_limit_orders = 0
profit = []

for i in range(1,len(local_max_index)):

  #Permitindo apenas canais diários
  if (df[local_max_index[i-1]:(local_max_index[i-1]+1)].index.day[0] != df[local_max_index[i]:(local_max_index[i]+1)].index.day[0]):
    continue

  firstPoint = df[local_max_index[i-1]:(local_max_index[i-1]+1)].high.values[0]
  secondPoint = df[local_max_index[i]:(local_max_index[i]+1)].high.values[0]

  barDistance_pips = firstPoint - secondPoint
  barDistance_range = local_max_index[i] - local_max_index[i-1]

  j=0
  calculatingPullbackEntry = False
  ordersTotal = False
  limitOrderPrice = 0
  orderRunner=0

  for close in df[(local_max_index[i]+1):len(df)].close.values:
    j += 1
    #Não pode ultrapassar horário limite
    if ((df[(local_max_index[i]+j):(local_max_index[i]+1+j)].index.hour[0]) >= trade_hour_threshold and 
    (df[(local_max_index[i]+j):(local_max_index[i]+1+j)].index.minute[0]) > trade_minute_treshold):
      break

    lineVariation = (barDistance_pips*(barDistance_range+j))/barDistance_range
    breakOutLine = firstPoint - lineVariation

    #Verificando breakout
    if(close > breakOutLine and calculatingPullbackEntry == False):
      calculatingPullbackEntry = True
      j += (n_bars_validation-2)

    #Atualizando limit order (pullback validado)
    elif(calculatingPullbackEntry == True):
      if ordersTotal == True:
        orderRunner +=1
        limitOrderPrice = breakOutLine + extra_range_entry
        limitOrderPrice = round_limit_order_price(limitOrderPrice)

        #Se ativar ordem, cai aqui
        if (df[(local_max_index[i]+j):(local_max_index[i]+j+1)].low.values[0] <= limitOrderPrice
            and df[(local_max_index[i]+j):(local_max_index[i]+j+1)].rsi.values[0] > 20):
          df['local_entry'][(local_max_index[i]+j):(local_max_index[i]+j+1)] = 1
          df['entry_value'][(local_max_index[i]+j):(local_max_index[i]+j+1)] = limitOrderPrice
          
          operation_timer = 0
          for k in df[(local_max_index[i]+j+1):(local_max_index[i]+j+operation_time+1)].i:
            operation_timer +=1
            stop_gain = limitOrderPrice+stop_gain_distance
            stop_loss = limitOrderPrice-stop_loss_distance

            if df[k:k+1].high.values >= stop_gain:
              df['exit'][k:k+1] = 1
              df['exit_value'][k:k+1] = stop_gain
              profit.append((stop_gain_distance*value_per_pip*trade_volume)-(order_tax*2*trade_volume))
              break

            elif df[k:k+1].low.values <= stop_loss:
              df['exit'][k:k+1] = 1
              df['exit_value'][k:k+1] = stop_loss
              profit.append(((-stop_loss_distance)*value_per_pip*trade_volume)-(order_tax*2*trade_volume))
              break

            if operation_timer == operation_time:
              df['exit'][k:k+1] = 1
              df['exit_value'][k:k+1] = df[k:k+1].close.values
              profit.append(((df[k:k+1].close.values[0] - limitOrderPrice)*value_per_pip*trade_volume)-(order_tax*2*trade_volume))
              break
          break

        #Ordem limite precisa ser acionada em menos de x minutos
        if (orderRunner > order_to_be_filled_threshold):
          ignored_limit_orders += 1
          break

      #Validando pullback (breakout validado)
      elif (ordersTotal == False):

        if(df[(local_max_index[i]+j):(local_max_index[i]+j+1)].low.values[0] > breakOutLine):
          #Abrindo ordem limite
          limitOrderPrice = breakOutLine + extra_range_entry
          limitOrderPrice = round_limit_order_price(limitOrderPrice)
          ordersTotal = True 
        else:
          #Ignorando pullback
          ignored_pullbacks += 1
          break

profit = pd.DataFrame(profit)
profit.columns = profit.columns.map(str)
profit.rename(columns={'0':'profit_per_trade'}, inplace=True)
profit[['cum_sum']] = profit['profit_per_trade'].cumsum()
profit['positive_trades'] = np.where(profit.profit_per_trade > 0, profit.profit_per_trade, 0)
profit['negative_trades'] = np.where(profit.profit_per_trade <= 0, profit.profit_per_trade, 0)

payoff = round(abs(profit['positive_trades'].mean()/profit['negative_trades'].mean()),2)
winrate = len(profit[profit.positive_trades > 0])/len(profit)

#%%
#Exibindo evolução do patrimonio sobre o acumulado de operações
profit["cum_sum"].plot()
#%%
#Informações adicionais

print("Período:", df[0:1].index[-1].strftime('%Y-%m-%d %X'), "-", df[(len(df)-1):len(df)].index[0].strftime('%Y-%m-%d %X'),
      "\nSaldo líquido:", round(profit.cum_sum.iloc[-1],2), "reais",
      "\nNúmero de entradas:", len(profit),
      "\nTaxa de acerto:", round(winrate,2),
      "\nPayoff:", payoff,
      "\nMédia de lucro por operação:", round(profit.profit_per_trade.mean(),2),
      "\nDesvio padrão:", round(profit.profit_per_trade.std(),2),
      "\nNúmero de pullbacks ignorados:", ignored_pullbacks,
      "\nNúmero de ordens limite ignorados:", ignored_limit_orders,
      sns.histplot(profit.profit_per_trade, bins=15))

# %%
#Plotando um exemplo da amostra: exibindo max locais, entradas e saídas
dateBegin =  datetime(2022,4,12,13,40)
dateEnd = datetime(2022,4,12,16,0)

df2 = df[(df.index >= dateBegin) & (df.index <= dateEnd)].copy()

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
                    vertical_spacing=0.03, subplot_titles=('OHLC - 1 min timeframe', 'Volume'), row_width=[0.2, 0.7])
                    
fig.add_trace(go.Candlestick(x=df2.index, open=df2['open'], high=df2['high'], low=df2['low'], close=df2['close'], name="Candle"), row=1, col=1)
fig.add_trace(go.Scatter(x=df2[df2["local_max"] == 1].index, y=df2[df2["local_max"] == 1]["high"], name="Topo Local", mode="markers", marker_color="cyan", marker_symbol="x", marker_size=15, opacity=0.5), row=1, col=1)
fig.add_trace(go.Scatter(x=df2[df2["local_entry"] == 1].index, y=df2[df2["local_entry"] == 1]["entry_value"], name="Compra", mode="markers", marker_color="green", marker_symbol="circle", marker_size=15, opacity=0.5), row=1, col=1)
fig.add_trace(go.Scatter(x=df2[df2["exit"] == 1].index, y=df2[df2["exit"] == 1]["exit_value"], name="Venda", mode="markers", marker_color="red", marker_symbol="square", marker_size=15, opacity=0.5), row=1, col=1)

fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=700)
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[18, 9], pattern="hour")])

fig.show()
# %%
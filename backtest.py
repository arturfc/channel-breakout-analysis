import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import seaborn as sns
import ta
import time
import os
from scipy.signal import argrelextrema
from datetime import datetime

def round_limit_order_price(limitOrderPrice):
  decimal_value = limitOrderPrice % 1

  if decimal_value > 0.25 and decimal_value <= 0.5:
    limitOrderPrice = float(round(limitOrderPrice) + 0.5)
  elif decimal_value > 0.5 and decimal_value <= 0.75:
    limitOrderPrice = float(round(limitOrderPrice) - 0.5)
  else:
    limitOrderPrice = float(round(limitOrderPrice))

  return limitOrderPrice

def calculateChannelResistance(barDistance_pips, current_position, barDistance_range, first_point_value):
  lineVariation = (barDistance_pips*(barDistance_range+current_position))/barDistance_range
  channelResistance = first_point_value - lineVariation
  return channelResistance

#variáveis extras - dados do ativo WDO
value_per_pip = 10 #valor em R$ por variação de 1 ponto
order_tax = 1.33 #custo em R$ por ordem enviada
trade_hour_threshold = 17 #horário limite de trade 
trade_minute_threshold = 0 #minuto limite de trade
trade_volume = 1 #número de lotes/contratos
order_to_be_filled_threshold = 30 #tempo de limite para consumir a ordem existente
#

#Mais variáveis
#Armazenando possíveis combinações de variáveis
stop_gain_distance_arr = np.arange(start=4, stop=17, step=6)
stop_loss_distance_arr = np.arange(start=4, stop=17, step=6)

enable_RSI_arr = np.array([1, 0])
rsi_period_arr = np.array([9,10,11]) 
rsi_value_arr = np.array([30,50,70]) 

operation_time_arr = np.arange(start=10, stop=31, step=10)      
n_bars_validation_arr = np.arange(start=5, stop=16, step=5)
extra_range_entry_arr = np.array([0, 1])

combination_arr = np.array(np.meshgrid(
    stop_gain_distance_arr, 
    stop_loss_distance_arr, 
    enable_RSI_arr,
    rsi_period_arr,
    rsi_value_arr,
    operation_time_arr,
    n_bars_validation_arr,
    extra_range_entry_arr,
)).T.reshape(-1,8)

df_variables = pd.DataFrame(combination_arr, columns=[
    "stop_gain_distance",
    "stop_loss_distance",
    "RSI_enable",
    "RSI_period",
    "RSI_value",
    "operation_time",
    "n_bars_validation",
    "extra_range_entry"
    ])

df_rates = pq.ParquetFile("WDO_dados.parquet").read().to_pandas()
df_rates.time = pd.to_datetime(df_rates.time, format="%Y-%m-%d %H:%M:%S")
df_rates = df_rates.reset_index()

del df_rates["tick_volume"]
del df_rates["spread"]
del df_rates["real_volume"]
del df_rates["index"]

df_results = pd.DataFrame(columns=['Saldo Líquido','Número de entradas', 'Taxa de acerto','Payoff','Média de lucro por operação',"Desvio padrão"])
#
t0 = time.time()
#Backtest completo contendo as possíveis combinações

for var_combination in combination_arr:
    #Alimentando variáveis
    stop_gain_distance = var_combination[0] #distância do stop gain em pontos
    stop_loss_distance = var_combination[1] #distância do stop loss em pontos
    enable_RSI = var_combination[2] #0 ou 1 para habilitar RSI
    rsi_period = var_combination[3] #período RSI
    rsi_value = var_combination[4] #valor RSI
    operation_time = var_combination[5] #duração máxima de uma operação
    n_bars_validation = var_combination[6] #numero de barras para validar um pullback
    extra_range_entry = var_combination[7] #range para entrar adiantado na operação

    df = df_rates

    #criando local_max
    local_max_index = np.array(argrelextrema(df.high.values, np.greater, order=15, mode='wrap')[0])

    df['local_max'] = 0
    df.loc[local_max_index, 'local_max'] = 1

    #Obtendo index do local_high anterior
    prev_high_index_array = df.iloc[df.local_max.values == 1].index.values
    prev_high_index_array = np.roll(prev_high_index_array, 1)
    prev_high_index_array[0] = 0

    df['prev_high_index'] = 0
    df.loc[local_max_index, 'prev_high_index'] = np.where(
      df.loc[df.local_max.values == 1].local_max.values,
      prev_high_index_array, 0
      )

    #filtrando limite máximo de pesquisa
    total_min_threshold = trade_hour_threshold*60 + trade_minute_threshold

    daily_min = df.iloc[local_max_index].time.values.astype('datetime64[m]') - df.iloc[local_max_index].time.dt.normalize().values.astype('datetime64[m]')
    search_index_array = total_min_threshold - daily_min 

    df['search_index'] = 0
    df.loc[local_max_index, 'search_index'] = np.where(
      df.loc[df.local_max.values == 1].local_max.values,
      search_index_array, 0
      )

    #Habilitando apenas calculo de canais diários
    search_index_array = search_index_array.astype('int32')

    isNewDay_array = np.where(
      np.roll(search_index_array, 1) < search_index_array,
      True,False
      )

    isNewDay_array

    df['isNewDay'] = False
    df.loc[local_max_index, 'isNewDay'] = np.where(
      df.loc[df.local_max.values == 1].local_max.values,
      isNewDay_array, False
      )

    #filtrando candidatos a calculos
    conditions = [
      (df.iloc[local_max_index].local_max == 1) &
      (df.iloc[local_max_index].search_index > 0) &
      (df.iloc[local_max_index].isNewDay == False)
    ]

    filtered_index_array = np.select(conditions, [df.iloc[local_max_index].index], default = 0)
    filtered_index_array = filtered_index_array[filtered_index_array != 0]

    #criando array com informações necessárias para o cálculo de breakout
    useful_calc_info = df.iloc[filtered_index_array].apply(
      lambda x: (df.iloc[x.prev_high_index].high, x.high,x.prev_high_index, x.name, x.search_index),
      axis=1)

    useful_calc_info = useful_calc_info.to_numpy(dtype=object)

    #calculando breakout, pullbacks e entradas
    profit = np.array([])

    df['rsi'] = ta.momentum.rsi(df.close, rsi_period)

    df["local_entry"] = 0
    df["entry_value"] = 0

    df["exit"] = 0
    df["exit_value"] = 0

    for i in useful_calc_info:
      
     
      barDistance_pips = i[0] - i[1]
      barDistance_range = i[3] - i[2]

      starting_range = i[3]+1
      
      ending_range = i[3] + i[4]

      positionsTotal = False
      ordersTotal = False
      orderRunner=0
      

      #validando breakout
      for row in df.iloc[starting_range:ending_range+1].index.values:
        channelResistance = calculateChannelResistance(barDistance_pips, row-i[3], barDistance_range, i[0])

        #validando breakout
        if df.close.iloc[row] > channelResistance:
          row += n_bars_validation-1
          channelResistance = calculateChannelResistance(barDistance_pips, row-i[3], barDistance_range, i[0])
          
          #validando pullback
          if(df.low.iloc[row] > channelResistance):
            ordersTotal = True

          break

      #verificando possível entrada pelo pullback (considerando ordem enviada)
      if ordersTotal == True:
        for row in df.iloc[row+1:ending_range+1].index.values:

          orderRunner +=1

          channelResistance = calculateChannelResistance(barDistance_pips, row-i[3], barDistance_range, i[0])

          limitOrderPrice = channelResistance+extra_range_entry
          limitOrderPrice = round_limit_order_price(limitOrderPrice)

          #se ativar ordem, cai aqui
          if (df.low.iloc[row] <= limitOrderPrice):
            if (enable_RSI == 1):
                if (df.rsi.iloc[row] > rsi_value):
                    df.loc[row, 'local_entry'] = 1 
                    df.loc[row, 'entry_value'] = limitOrderPrice 
                    row_value = row
                    positionsTotal = True
                    break

            else:
                df.loc[row, 'local_entry'] = 1 
                df.loc[row, 'entry_value'] = limitOrderPrice 
                row_value = row
                positionsTotal = True
                break
            
          elif orderRunner > order_to_be_filled_threshold:
            break
        

      if (positionsTotal == True):
        operation_timer = 0 
        for row in df.iloc[row+1:(row+1+operation_time)].index.values:
          operation_timer += 1

          stop_gain = limitOrderPrice+stop_gain_distance
          stop_loss = limitOrderPrice-stop_loss_distance

          if df.high.iloc[row] >= stop_gain:
            df.loc[row, 'exit'] = 1 
            df.loc[row, 'exit_value'] = stop_gain
            profit = np.append(profit, (stop_gain_distance*value_per_pip*trade_volume)-(order_tax*2*trade_volume))
            break

          elif df.low.iloc[row] <= stop_loss:
            df.loc[row, 'exit'] = 1
            df.loc[row, 'exit_value'] = stop_loss
            profit = np.append(profit, ((-stop_loss_distance)*value_per_pip*trade_volume)-(order_tax*2*trade_volume))
            break

          elif operation_timer == operation_time:
            df.loc[row, 'exit'] = 1
            df.loc[row, 'exit_value'] = df.close.iloc[row]
            profit = np.append(profit, ((df.close.iloc[row] - limitOrderPrice)*value_per_pip*trade_volume)-(order_tax*2*trade_volume))
            break


    profit = pd.DataFrame(profit, columns=["profit_per_trade"])
    profit[['cum_sum']] = profit['profit_per_trade'].cumsum()
    profit['positive_trades'] = np.where(profit.profit_per_trade > 0, profit.profit_per_trade, 0)
    profit['negative_trades'] = np.where(profit.profit_per_trade <= 0, profit.profit_per_trade, 0)


    results_array = np.array(
        [round(profit.cum_sum.iloc[-1],2),
        len(profit),
        round(len(profit[profit.positive_trades > 0])/len(profit),2),
        round(abs(profit['positive_trades'].mean()/profit['negative_trades'].mean()),2),
        round(profit.profit_per_trade.mean(),2),
        round(profit.profit_per_trade.std(),2)]
        )

    df_results = df_results.append(pd.DataFrame(results_array.reshape(1,-1), columns=list(df_results)), ignore_index=True)

#loop termina aqui
t1 = time.time()
total = t1-t0
print("Tempo total de execução",total)

file_list = ['Resultados.csv', 'Variaveis.csv']
for file in file_list:
  if(os.path.exists(file) and os.path.isfile(file)):
    os.remove(file)

df_results.to_csv('Resultados.csv')
df_variables.to_csv('Variaveis.csv')

#printando as 5 melhores e piores combinações
best_variable_index_arr = df_results.sort_values('Saldo Líquido', ascending=False).index
worst_variable_index_arr = df_results.sort_values('Saldo Líquido').head(5).index

print(
    "\nMelhores 5 resultados:\n",
    df_results.iloc[best_variable_index_arr[0:5]],
    #"\nConfiguração dos 5 melhores resultados:\n",
    #df_variables.iloc[best_variable_index_arr[0:5]],
    "\n\nPiores 5 resultados:\n",
    df_results.iloc[worst_variable_index_arr[0:5]]
    #"\nConfiguração dos 5 melhores resultados:\n",
    #df_variables.iloc[worst_variable_index_arr[0:5]]
)

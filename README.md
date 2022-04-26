# Channel Resistance Breakout + Pullback Analysis

<div align="center">
<img src="https://user-images.githubusercontent.com/20324343/165372057-8e91e766-0ca4-4e07-bb92-3b79e8ed725d.png" width="700px" />
</div>

Este projeto apresenta um estudo de análise quantitativa que verifica a quebra (breakout) da linha de **resistência** de canais de **alta** ou de **baixa**. Caso haja um breakout, o programa irá calcular a existência de um pullback (retorno do preço à resistência do canal) e validar a simulação de uma ordem limite, que atualiza a cada surgimento de uma nova barra (candle), até esta ordem seja ativada ou o tempo de espera expire. O objetivo principal é produzir backtests massivos para descobrir quais configurações de inputs possuem os melhores resultados para **saldo líquido**, **taxa de acerto**, **payoff**, **média de lucro por operação** e **desvio padrão**, tudo com o menor número de entradas possível. 

O estudo foi realizado utilizando dados históricos do mini dolar - WDO@N em timeframe de 1 minuto, dentro do período de 2021-07-26 16:51:00 até 2022-04-18 17:59:00 . O código, de forma geral, visa executar as seguintes tarefas:

- Encontrar máximas locais e interligá-los para formação de canais de tendências;
- Busca do momento em que o preço quebra a linha de resistência de um canal;
- Validação de um possível pullback posterior do breakout;
- Simulação de envio de ordem que dura por n minutos ou até a sua ativação;
- Saída da operação de acordo com os inputs a serem configurados.

Para que o canal de tendência seja inicialmente aprovado para os cálculos, ele precisa passar pelas seguintes condições:

- Os dois pontos que representam máximas locais precisam ser intradiários;
- Se o segundo ponto que representa máxima local for gerado após o horário de limite de trade configurado, ele será ignorado

# Arquivos do projeto

- WDO_dados.parquet possui as informações da série histórica do mini dolar, coletadas a partir do MetaTrader 5
- [backtest.py](backtest.py) irá aproximadamente 3000 diferentes conbinações de inputs para procurar as melhores e piores configurações para a estratégia
- Os arquivos .csv correspondem ao output gerado pelo backtest, provendo as informações coletadas como os próprios nomes do arquivo sugerem
- [main.py](main.py) contém o passo a passo, de forma a guiar o leitor, a entender como foi construído a lógica do código
- [main.ipynb](main.ipynb) representa o mesmo código que main.py, porém de forma mais interativa, a fim de ilustrar as saídas/imagens geradas

# Code comparison file

O propósito dos arquivos contidos nela é, exibir o tempo de execução dos códigos implementados existentes. Os mesmos possuem as mesmas lógicas de programação, porém foram escritas de maneiras diferentes:

- goodCode.py representa o código que aproveita os métodos de vetorização providos do pandas e numpy, .apply e lambda, a fim de agilizar a velocidade código.
- badCode.py representa o código intencionalmente mal feito, possuindo maior uso de For Loop

Tempo de execução:

- goodCode.py: 1.8712852001190186 segundos
- badCode.py: 18.624610424041748 segundos
---

## Contributors
- Artur Fernandes e Cunha <arturfernandescunha@gmail.com>

---

# License & copyright

© Artur Fernandes e Cunha

Licensed under the [MIT License](LICENSE).

---
Exemplo da imagem do título em plot, contendo as amostras específicas:
<div align="center">
<img src="https://user-images.githubusercontent.com/20324343/165372170-63cc7eb7-79c0-4520-82b6-7f7d68993bcd.png" width="1000px" />
</div>

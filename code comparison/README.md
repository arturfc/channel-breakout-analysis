# Channel Resistance Breakout + Pullback Analysis - extra codes

O propósito aqui é exibir apenas printar o tempo de execução dos códigos existentes nesta pasta. Os mesmos possuem as mesmas lógicas de programação, porém foram escritas de maneiras diferentes:

- goodCode.py representa o código que aproveita os métodos de vetorização providos do pandas e numpy, .apply e lambda, a fim de agilizar a velocidade código.
- badCode.py representa o código intencionalmente mal feito, possuindo maior uso de For Loop


Tempo de execução:
- goodCode.py: 1.871 segundos
- badCode.py: 18.625 segundos

\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
% The preceding line is only needed to identify funding in the first footnote. If that is unneeded, please comment it out.
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}
\begin{document}

\title{Análise Comparativa de Algoritmos de Aprendizado de Máquina na Predição do Crescimento do Trânsito Urbano a partir de Dados Populacionais\\

% \thanks{Identify applicable funding agency here. If none, delete this.}
}

\author{\IEEEauthorblockN{Alexandre Marques Tortoza Canoa}
\IEEEauthorblockA{\textit{Escola Politécnica} \\
\textit{Pontifícia Universidade Católica do Paraná (PUCPR))}\\
Curitiba, PR, Brasil \\
a.marquestortoza@gmail.com}
}

\maketitle

\begin{abstract}
O agravamento do congestionamento urbano, impulsionado pelo crescimento populacional e pela expansão da frota de veículos, afeta diretamente a economia, o meio ambiente e o bem-estar social. Para estimar os níveis médios de tráfego nas regiões das cidades, foi aplicada uma abordagem preditiva que combina dados demográficos e históricos de trânsito. Após etapas de pré-processamento e codificação, foram testados diversos algoritmos de aprendizado supervisionado. Os modelos baseados em árvores apresentaram melhor desempenho, com destaque para o Random Forest. A população total foi a variável mais relevante, seguida por fatores temporais e espaciais. A integração entre dados populacionais e padrões de tráfego mostrou-se eficaz para apoiar o planejamento urbano e a gestão inteligente da mobilidade.
\end{abstract}

\begin{IEEEkeywords}
component, formatting, style, styling, insert
\end{IEEEkeywords}

\section{Introdução}

O congestionamento urbano é um dos principais desafios enfrentados pelas grandes cidades contemporâneas. Intensificado pelo crescimento populacional acelerado e pela expansão desordenada da frota de veículos, esse fenômeno compromete diretamente a qualidade de vida, a eficiência econômica e a sustentabilidade ambiental. Estudos indicam que, em países desenvolvidos, os prejuízos causados pelo tráfego intenso somam bilhões de dólares por ano, refletindo em tempo perdido, aumento do consumo de combustível, elevação dos níveis de poluição e maior incidência de acidentes de trânsito (Shah et al., 2023). Na América Latina, o cenário é agravado pelo crescimento urbano desordenado e pela rápida motorização (TRB, 1981).
\subsection{Motivação}

Mitigar os impactos econômicos, sociais e ambientais do congestionamento é uma prioridade nas grandes metrópoles. A poluição, os atrasos nas viagens e os custos elevados de transporte exigem soluções eficazes e escaláveis. Antecipar cenários críticos de tráfego é essencial para otimizar a mobilidade e apoiar decisões em tempo real. Técnicas de aprendizado de máquina, aplicadas a dados históricos de tráfego, permitem identificar padrões e prever situações de congestionamento, viabilizando ações preventivas como ajustes semafóricos, orientação aos motoristas e melhorias no transporte público. Diversas abordagens têm demonstrado o potencial dessas técnicas para aumentar a eficiência do sistema viário (Shah et al., 2023).
\subsection{Problema}

Nas áreas urbanas, especialmente em grandes centros, o crescimento populacional e a motorização acelerada intensificam os congestionamentos, gerando impactos diretos na população e na economia. Entre os principais efeitos estão o aumento da poluição atmosférica e sonora, o maior consumo de combustível, a perda de tempo nas viagens e o crescimento das taxas de acidentes e infrações. Nos Estados Unidos, o congestionamento é apontado como uma ameaça ao desempenho econômico, enquanto na América Latina o problema se agrava com o crescimento desordenado das cidades.
\subsection{Objetivo}

Esta pesquisa busca prever, com antecedência, o nível de congestionamento em vias urbanas da cidade de São Paulo, utilizando séries temporais de volume de veículos. O objetivo é modelar a variação do fluxo ao longo do tempo, identificar padrões recorrentes e antecipar os momentos e locais mais propensos a congestionamentos, oferecendo suporte ao planejamento e à gestão inteligente do trânsito.
\subsection{Artefato}

Será desenvolvido um sistema de aprendizado de máquina treinado com dados históricos de tráfego e rótulos de congestionamento. Após o treinamento, o modelo receberá dados atuais como entrada e estimará o nível de congestionamento de curto prazo, disponibilizando as previsões em uma interface voltada ao monitoramento e à tomada de decisão antecipada.
\subsection{Solução}

A aplicação de tecnologias inteligentes no trânsito urbano tem ganhado destaque no Brasil. Em Joinville (SC), pesquisadores criaram um sistema web que integra YOLOv3 ao OpenCV com aceleração via CUDA, capaz de detectar, contar e classificar veículos em vídeos captados por drones, gerando relatórios e dashboards para gestores públicos (FARIAS; NUNES, 2021). Na Universidade Nove de Julho, um estudo comparou algoritmos de subtração de fundo (Background Subtraction — BGS) para identificar veículos e pedestres, concluindo que sua eficácia aumenta quando combinados com técnicas de machine learning ou deep learning (OLIVEIRA, 2019). Já na PUC Goiás, redes neurais artificiais baseadas no modelo Intelligent Driver Model (IDM) foram utilizadas para ajustar dinamicamente os ciclos semafóricos com base em dados reais de tráfego, resultando em leve redução no tempo de espera e melhora na fluidez do trânsito (COSTA; SILVA, 2024).

\section{Estado da Arte}

O congestionamento urbano é um problema persistente que acompanha a expansão das cidades desde o século XX, impulsionado pelo aumento da frota de veículos e pela incapacidade das infraestruturas viárias de atender à crescente demanda (DOWNS, 1992; LITMAN, 2004; VIANNA; YOUNG, 2017). Esse fenômeno gera perdas significativas de tempo produtivo, elevação dos níveis de poluição e impactos econômicos expressivos, especialmente em regiões metropolitanas como São Paulo e Rio de Janeiro, que figuram entre as mais congestionadas do mundo (VALE, 2018).

Historicamente, diferentes estratégias foram adotadas para mitigar esse problema. Nas décadas de 1950 e 1960, predominou o modelo rodoviarista, com foco na ampliação da malha viária e incentivo ao transporte individual (VIANNA; YOUNG, 2017). A partir dos anos 1990, o debate passou a incorporar questões sociais, como o tempo de deslocamento casa-trabalho, revelando desigualdades entre classes sociais — trabalhadores de baixa renda chegam a gastar até 20\% mais tempo em viagens diárias do que os mais ricos (PEREIRA; SCHWANEN, 2013).

O congestionamento, no entanto, não decorre apenas da quantidade de veículos. Fatores como comportamento dos motoristas, condições das vias, clima, incidentes, obras e eventos especiais também influenciam diretamente o fluxo viário (JHA; ALBERT, 2021). Esses elementos podem ser agrupados em causas recorrentes e não recorrentes, ampliando a complexidade da gestão urbana e abrindo espaço para soluções computacionais baseadas em variáveis ambientais, infraestruturais e comportamentais.

Com o avanço das tecnologias digitais, gestores urbanos passaram a contar com ferramentas mais sofisticadas. A era do Big Data e da Inteligência Artificial trouxe novas possibilidades para o monitoramento e controle do tráfego. Soluções baseadas em aprendizado de máquina vêm sendo aplicadas em diferentes contextos, como a previsão de velocidades em rodovias (ZECHIN et al., 2022), o controle adaptativo de semáforos (SIQUEIRA FILHO, 2024) e o monitoramento de veículos e pedestres por meio de visão computacional (BARBOZA, 2023; MAUS et al., 2021).

A visão computacional, em especial, tem ampliado as capacidades de análise em tempo real. Barboza (2023) destaca o uso de algoritmos de subtração de fundo (BGS) para detectar veículos e pedestres, que, embora limitados quando utilizados isoladamente, tornam-se eficazes quando combinados com técnicas de aprendizado de máquina e dispositivos IoT. Esses sistemas são promissores para o desenvolvimento de semáforos adaptativos e soluções inteligentes em cidades conectadas.

Maus et al. (2021) apresentam uma aplicação que utiliza YOLOv3, Darknet e OpenDataCam para detectar e classificar veículos em tempo real a partir de imagens captadas por drones. Essa abordagem permite não apenas a contagem de veículos, mas também a análise de rotas, fluxos e padrões de tráfego, gerando relatórios e dashboards que auxiliam gestores na tomada de decisão.

Além disso, a integração entre visão computacional e aprendizado profundo tem sido explorada para otimizar o tempo semafórico. Siqueira Filho (2024) propõe o uso de redes neurais artificiais para ajustar dinamicamente os ciclos dos semáforos conforme a demanda do tráfego, substituindo o controle fixo por sistemas inteligentes capazes de reduzir congestionamentos em interseções críticas e melhorar a fluidez urbana.

Diante desse panorama, a presente pesquisa se insere na tradição de busca por soluções para um problema antigo, agora potencializado pelas ferramentas da computação moderna. A proposta é explorar técnicas de aprendizado de máquina como alternativa para mitigar os impactos do congestionamento, utilizando dados históricos e sistemas inteligentes capazes de se adaptar dinamicamente às condições reais do tráfego urbano.

\section{Metodologia}
A metodologia adotada nesta pesquisa foi estruturada em quatro etapas principais, coleta de dados, tratamento e preparação, testes com algoritmos de machine learning e análise dos resultados. O pipeline seguido está ilustrado na Figura 1.

\begin{figure}[htbp]
\centering
\includegraphics[width=0.4\textwidth]{pipeline.png}
\caption{Pipeline da metodologia}
\label{fig:pipeline}
\end{figure}

Inicialmente, partimos de um dataset referente ao trânsito da cidade de São Paulo. Para enriquecer a análise e estabelecer uma correlação entre o comportamento do trânsito e o crescimento populacional da capital, buscamos dados complementares no portal https://dadosabertos.sp.gov.br. De lá, extraímos um segundo dataset contendo informações sobre a população de São Paulo ao longo dos anos.

Na segunda etapa, realizamos a limpeza e padronização dos dados. Os datasets apresentavam ruídos, campos fora de padrão e diferentes codificações de texto (encodings), o que exigiu um processo cuidadoso de tratamento para garantir a compatibilidade e integridade das informações. Após a unificação dos dados, com as variáveis devidamente estruturadas, avançamos para a etapa de testes com algoritmos de aprendizado de máquina.

Os testes foram conduzidos com diferentes modelos, e os resultados foram armazenados para posterior comparação, tanto em termos de desempenho quanto de qualidade preditiva. Essa abordagem permitiu avaliar a eficácia dos algoritmos frente aos dados tratados e entender melhor a relação entre o crescimento populacional e o impacto no trânsito da cidade.

\subsection{Datasets}
O ponto de partida da pesquisa foi a seleção de um dataset sobre o trânsito da cidade de São Paulo. Este conjunto de dados se mostrou especialmente relevante por conter informações como dia, hora, região e, principalmente, o nível de congestionamento, variável central para os objetivos do estudo. Algumas colunas adicionais estavam presentes, mas foram descartadas por não contribuírem diretamente para a análise proposta.

Para estabelecer uma correlação entre o comportamento do trânsito e o crescimento populacional da cidade, foi incorporado um segundo dataset, obtido por meio do portal dadosabertos.gov.sp Este conjunto trazia dados populacionais da capital paulista ao longo dos anos, incluindo as variáveis ano, distrito, sexo, faixa etária e população total.

Apesar da riqueza informacional, esse segundo dataset apresentou desafios técnicos. Havia ruídos nos dados, campos fora de padrão e codificações de texto distintas (encodings), o que exigiu um esforço adicional para uniformização e tratamento adequado. Um ponto específico que merece destaque é a estrutura da variável idade, que estava agrupada em faixas etárias (por exemplo, "10 a 14 anos", "15 a 19 anos"). Essa segmentação dificultou a filtragem precisa de indivíduos com idade mínima para conduzir veículos (a partir dos 18 anos), sendo possível apenas considerar faixas a partir dos 20 anos, o que introduz um pequeno desvio na análise.

A aquisição dos dados, portanto, envolveu não apenas a seleção criteriosa dos conjuntos mais relevantes, mas também a antecipação dos desafios que seriam enfrentados nas etapas seguintes de tratamento e modelagem.

\subsection{limpeza e preparacao}

Para assegurar a consistência e a qualidade dos dados utilizados na modelagem preditiva, foram desenvolvidos dois pipelines automatizados de pré-processamento, um voltado ao tratamento dos dados populacionais e outro dedicado ao conjunto de dados de congestionamentos de tráfego. Ambos os processos tiveram como objetivo padronizar formatos, remover inconsistências e preparar os dados para a etapa de integração e posterior aplicação de algoritmos de aprendizado de máquina.

O conjunto populacional passou por um tratamento mais detalhado, executado pelo script python . Inicialmente, todos os campos textuais foram normalizados por meio da remoção de acentuação e da padronização de espaços, garantindo uniformidade entre distritos e categorias. Em seguida, foi aplicado um filtro para manter apenas as faixas etárias com idade igual ou superior a 20 anos, de por conta de limitações do dataset, mais diretamente relacionada aos padrões de deslocamento urbano. Na sequência, foi criada uma nova variável denominada região, obtida a partir do mapeamento de cada distrito para uma das cinco grandes áreas geográficas de São Paulo: norte, sul, leste, oeste e centro. Esse mapeamento foi implementado por meio de um dicionário abrangendo os 96 distritos oficiais do município, permitindo uma agregação coerente com as divisões espaciais utilizadas nos dados de tráfego. Por fim, distritos que não possuíam correspondência foram identificados e registrados em um arquivo auxiliar para verificação manual, garantindo a completude do mapeamento antes da exportação final dos dados, que foram salvos em formato CSV com codificação UTF-8 padronizada.

O conjunto de dados de tráfego, processado pelo script python, exigiu um tratamento voltado principalmente à validação e à consistência das informações temporais e geográficas. As etapas iniciais incluíram a normalização dos campos textuais referentes a vias, regiões e expressways, removendo acentuação e espaços redundantes. Em seguida, foi realizada a validação das variáveis de data e hora, assegurando que apenas registros com formato temporal correto fossem mantidos. O campo de tamanho do congestionamento também passou por verificação, sendo eliminados valores negativos, nulos ou não numéricos. As regiões foram padronizadas para um formato em letras minúsculas e comparadas a uma lista de referência contendo as mesmas categorias usadas no dataset populacional, garantindo a compatibilidade entre as bases. Registros duplicados foram identificados e removidos, e, ao final, foi gerado um relatório estatístico que resumiu a distribuição dos congestionamentos por região, além de medidas descritivas como média, mediana e desvio padrão do tamanho dos congestionamentos.

Ambos os conjuntos de dados resultantes foram padronizados quanto à codificação e ao delimitador, de forma a assegurar compatibilidade durante o processo de integração. Essa padronização foi fundamental para que os datasets pudessem ser combinados por meio das variáveis região e ano, possibilitando análises conjuntas sobre o impacto da densidade populacional no comportamento do tráfego e permitindo que as etapas seguintes de modelagem preditiva fossem conduzidas de maneira consistente e reprodutível.








\begin{thebibliography}{00}
\bibitem{b1} R. S. Barboza, ``Visão computacional: estudo comparativo de algoritmos de subtração de fundo aplicados em soluções para o gerenciamento de tráfego urbano de veículos e pedestres,'' M.Sc. dissertation, Programa de Mestrado em Cidades Inteligentes e Sustentáveis, Univ. Nove de Julho, São Paulo, Brazil, 2023.

\bibitem{b2} A. Downs, \textit{Stuck in traffic: coping with peak-hour traffic congestion}. Washington, D.C.: Brookings Institution, 1992.
b
\bibitem{b3} K. Jha and L. Albert, ``Congestion pie chart for different sources of congestion,'' Technical Memorandum, Support for Urban Mobility Analysis (SUMA), FHWA Pooled Fund Study, 2021.

\bibitem{b4} T. Litman, ``Transit price elasticities and cross-elasticities,'' \textit{Journal of Public Transportation}, vol. 7, no. 2, pp. 37–58, 2004.

\bibitem{b5} A. V. Maus \textit{et al.}, ``Contagem e classificação de veículos por visão computacional,'' in \textit{Proc. XII Computer on the Beach}, Joinville, Brazil: SENAI, 2021.

\bibitem{b6} R. H. M. Pereira and T. Schwanen, ``Tempo de deslocamento casa-trabalho no Brasil (1992–2009): diferenças entre regiões metropolitanas, níveis de renda e sexo,'' Brasília: IPEA, 2013.

\bibitem{b7} L. A. Siqueira Filho, ``Redes neurais aplicadas a semáforos de trânsito,'' B.Sc. thesis, Curso de Ciência da Computação, Pontifícia Univ. Católica de Goiás, Goiânia, Brazil, 2024.

\bibitem{b8} R. C. C. Vale, ``The welfare costs of traffic congestion in São Paulo Metropolitan Area,'' M.Sc. dissertation, Programa de Mestrado em Economia Aplicada, Univ. de São Paulo, Ribeirão Preto, Brazil, 2018.

\bibitem{b9} G. S. B. Vianna and C. E. F. Young, ``In search of lost time: an estimate of the product losses in traffic congestion in Brazil,'' in A. T. Dantas, W. Koziol, and R. Siuda-Ambroziak, Eds., \textit{Brazil-Poland: focus on environment}. Rio de Janeiro: UERJ/Nucleas; Warsaw: Univ. of Warsaw/CESLA, 2017.

\bibitem{b10} D. Zechin, M. B. Amaral, and H. B. B. Cybis, ``Previsão de velocidades de tráfego com rede neural LSTM encoder-decoder,'' \textit{Transportes}, vol. 30, no. 3, pp. 1–18, 2022.

\end{thebibliography}

\end{document}

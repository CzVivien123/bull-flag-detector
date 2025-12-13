\# Bull-flag detector (time-series)



Deep Learning házi projekt – bull/bear flag mintázatok felismerése idősorokban.



\## Projekt cél

A cél egy olyan modell készítése, ami:

\- felismeri a bull és bear flag mintázatokat,

\- megkülönbözteti a flag típusokat,

\- elkülöníti a nem-flag jellegű szakaszoktól.



Osztályok:

\- Bullish Normal

\- Bullish Wedge

\- Bullish Pennant

\- Bearish Normal

\- Bearish Wedge

\- Bearish Pennant



\## Könyvtárstruktúra

```text

Bull\_flag\_detector/

├─ src/

├─ notebook/

├─ data/

├─ output/

├─ requirements.txt

├─ Dockerfile

└─ README.md

```



\## Bemeneti adatok

A `data/` mappába kerülnek:

\- CSV idősor fájl

\- Label Studio által generált JSON annotáció



\## Futtatás Dockerrel

\### Image build

```cmd

docker build -t bullflag:1.0 .

```



\## Konténer futtatása

```cmd

docker run --rm ^

&nbsp; -v "%cd%\\data:/data" ^

&nbsp; -v "%cd%\\output:/app/output" ^

&nbsp; bullflag:1.0

```



\## Megjegyzés

\- A `data/` és `output/` mappák nem részei a verziókövetésnek

\- A projekt célja egy reprodukálható, Dockerből futtatható pipeline


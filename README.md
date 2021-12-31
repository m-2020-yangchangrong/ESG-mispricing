# Code and data for *ESG Mispricing: A View from the Overnight Return*

[Xiaoqun Liu](https://) (Hainan University), [Changrong Yang](https://) (Hainan University), and [Youcong Chao](https://) (North China Institute of Aerospace Engineering)

The latest version of this code can be found at [https://github.com/m-2020-yangchangrong/ESG-mispricing](https://github.com/m-2020-yangchangrong/ESG-mispricing).

*Note*: this document is written in Github Flavored Markdown. It can be read by any text editor, but is best viewed with a GFM viewer.

## Code

All provided code has been tested with 3.9.7 and the packages listed in `requirements.txt`.

### Main analysis

We provide two Jupyter notebooks:

- `Main Analysis.ipynb`: Contains the code to replicate all tables of the paper.

## Data

This study employs several datasets:
1. Bloomberg for ESG socre (e.g., ESG score, E score, S score, G score).
2. CSMAR for Fama-French five factors (e.g., mkt_rf, smb, hml, rmw, cma).
3. CSMAR for daily stock observations (e.g., price, turnover).

### Stock sample

The file `fivefactor_yearly` contains the Fama-French five factors used in this study, with the following columns:
- `mkt_rf`: market factor.
- `smb`: size factor.
- `hml`: book-to-market ratio factor.
- `rmw`: profitability factor.
- `cma`: investment factor.
- `rf`: risk-free rate.

The file `yearly_indicator` contains the sample of stocks used in this study, with the following columns:
- `Stkcd`: stock code.
- `Trdyear`: date.
- `overnight_return`: overnight return for each stock i of each year t by accumulating corresponding daily overnight return.
- `size`: market capitalization.
- `BM`: book-to-market ratio.
- `ILLIQ`: Amihud (2002) illiquidity.
- `turnover`: yearly turnover.
- `ESG_score`: TAQ symbol.
- `E_score`: TAQ symbol.
- `S_score`: TAQ symbol.
- `G_score`: TAQ symbol.

The data is provided in single formats:
- Comma-separated values (CSV)


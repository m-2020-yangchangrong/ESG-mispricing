# Code and data for *The Pricing of ESG: Evidence from Overnight Return and Intraday Return*

[Xiaoqun Liu](https://github.com/LiuFinance) (Hainan University), 
[Changrong Yang](https://github.com/m-2020-yangchangrong) (Hainan University) and 
[Youcong Chao](https://) (North China Institute of Aerospace Engineering)

`Abstract` : By featuring the link of investor heterogeneity to the persistence of the 
overnight and intraday components of returns, we examine the ESG-overnight (intraday) 
alpha relation in the Chinese stock market. The empirical results present that ESG score 
has a significantly negative effect on the expected stock overnight returns in 
Fama-MacBeth regression. Consistently, given that the biggest market capitalization and 
the least illiquidity subsamples, the trading strategies by going long (short) the top (bottom) 
ESG quintile would yield negative profits. In addition, we conduct the implication of the 
ESG pricing by dividing the full sample into green stocks subsample and sin stocks subsample, 
and the empirical results present that the ESG pricing is pervasive of the green-type stocks. 
These conclusions verify the pricing of ESG, and support the conjecture that green stocks 
have lower expected returns because ESG investors value sustainability.

`Keywords`: ESG pricing; Overnight return; Trading strategy; Fama-MacBeth regression; Green stock

The latest version of this code can be found at [https://github.com/m-2020-yangchangrong/ESG-mispricing](https://github.com/m-2020-yangchangrong/ESG-mispricing).

*Note*: this document is written in Github Flavored Markdown. It can be read by any text editor, but is best viewed with a GFM viewer.



## Usage

- git clone https://github.com/m-2020-yangchangrong/ESG-mispricing.git
- cd ESG_mispricing
- pip install -r requirements.txt (optional)
- jupyter notebook

## Code

All provided code has been tested with python 3.9.7 and the packages listed in `requirements.txt`.

### Main analysis

We provide one Jupyter notebook :

- `Main Analysis.ipynb`: Contains the code to replicate all tables of the paper.

### Utils

We provide some python utils:

- `portfolios1D.py` for Univariate portfolio 
- `portfolios2D.py` for Two-variable portfolio
- `regression_demo.py` for regression output format

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
- `ESG_score`:  environmental, social, governance score.
- `E_score`: environmental score.
- `S_score`: social score.
- `G_score`: governance score.

The data is provided in single formats:
- Comma-separated values (CSV)

## Citation

Please cite our paper if you use this repo in your work

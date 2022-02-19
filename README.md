# Abstract 

Biophysical parameter retrieval using remote sensing has long been utilized for crop yield forecasting and economic practices. Remote sensing can provide information across a large spatial extent and in a timely manner within a season. Plant Area Index (PAI), Vegetation Water Content (VWC), and Wet-Biomass (WB) play a vital role in estimating crop growth and helping farmers make market decisions. Many parametric and non-parametric machine learning techniques have been utilized to estimate these parameters. A general non-parametric approach that follows a Bayesian framework is the Gaussian Process (GP). The parameters of this process-based technique are assumed to be random variables with a joint Gaussian distribution. The purpose of this work is to investigate Gaussian Process Regression (GPR) models to retrieve biophysical parameters of three annual crops utilizing combinations of multiple polarizations from C-band SAR data. RADARSAT-2 full-polarimetric images and in situ measurements of wheat, canola, and soybeans obtained from the SMAPVEX16 campaign over Manitoba, Canada, are used to evaluate the performance of these GPR models. The results from this research demonstrate that both the full-pol (HH+HV+VV) combination and the dual-pol (HV+VV) configuration can be used to estimate PAI, VWC, and WB for these three crops.

## ![giphy]<img src="https://user-images.githubusercontent.com/42670579/154797745-a6a1b320-372a-41c6-8171-3a7c78b8a146.gif" width="60" height="40">) Code avalaibility 

Codes for retrieving wheat and canola biophysical parameters utilizing gaussian process regression are availaible now !!!

Code for plotting temporal analysis of backscatter intensitites is availaible now !!!



## How to cite my work 

If you find our article useful kindly cite it using the following bibtex: 

@Article{rs14040934,
AUTHOR = {Ghosh, Swarnendu Sekhar and Dey, Subhadip and Bhogapurapu, Narayanarao and Homayouni, Saeid and Bhattacharya, Avik and McNairn, Heather},
TITLE = {Gaussian Process Regression Model for Crop Biophysical Parameter Retrieval from Multi-Polarized C-Band SAR Data},
JOURNAL = {Remote Sensing},
VOLUME = {14},
YEAR = {2022},
NUMBER = {4},
ARTICLE-NUMBER = {934},
URL = {https://www.mdpi.com/2072-4292/14/4/934},
ISSN = {2072-4292},
DOI = {10.3390/rs14040934}
}

# Urban Heat Island (UHI)


This repository is dedicated to investigating the UHI effect using CORDEX-CORE (0.22°) data.



It includes functionalities to identify urban a rural cells and assess UHI in large urban areas across the globe. 

## Repository Overview
### Key Components
1. **Morphological Dilation Function**: This function helps in selecting urban and rural cells using fixed variables (sftuf, orog, sftlf).

    - sftuf: Fraction of urban land use.
    - orog: Orography (elevation).
    - sftlf: Land-sea mask.

2. **Analyzing UHI Effect**: The UHI effect can be analyzed for all CORDEX-CORE domains using REMO and RegCM models for evaluation scenarios. The key variables for this analysis are:

    - tasmin: Minimum temperature.
    - tasmax: Maximum temperature.

3. **Cities.ipynb Notebook**: This Jupyter Notebook allows users to explore different cities and analyze the UHI effect in those areas.



# DarkEFT 
*Version 1.0 - 06/01/2020 -  L. Darme, S. A. R. Ellis, T. You*

DarkEFT is a python tool to obtain the constraints on fermion portal effective operators, based on existing and prospective experimental searches. The physics behind it has been described in the paper "Light Dark Sectors through the Fermion Portal", available at https://arxiv.org/abs/2001.01490 .
Its main features are:
 - A database of relevant analysis and limits, along with the relevant references and a small description.
 - Amplitudes for various relevant production and decay mechanisms for dark sector states within the effective field theory 
 - A set of tools to either recast the stored limits to the fermion portal case or create naive limits neglecting experimental efficiencies
 
*While using DarkEFT, please cite* [Light Dark Sectors through the Fermion Portal](https://arxiv.org/abs/2001.01490) *as well as the individual citation for each relevant experimental or projected/recasted limit used in your work*. Relevant individual citation can be found either in our paper, or in LimitsList.py or directly in the python through `yourLimitUsed.ref`.

### Requirements
The code has been written for python 3 and requires additionally
- numpy 
- scipy (for the numerical integrations of certain branching ratios and interpolation)
- matplotlib (to use the example plots, not required to obtain the limits)

### Basic start-up 
Just copy-paste one example from the Examples/ folder into the main `DarkEFT` folder and run it as e.g.:
```shell
python3 Example1.py
```
We have distributed five examples demonstrating the various uses of the code, as well as `Plot_Papers.py` which can be used to recreate the plots from 2001.01490. As an example, recasting the limits from the MiniBooNE collaboration from light dark matter scattering presented in https://arxiv.org/abs/1807.06137, for a electromagnetically-aligned vector operator and for a splitting of 25% between X_1 and X_2 can be done by
```python
import LimitsList as lim
geffem={"gu11":2/3.,"gd11":-1/3.,"gl11":-1.} 
xi_full,Lim_full= lim.miniboone_scattering.recast(0.25,geffem,"V")
```
### Structure

#### Main modules
DarkEFT typically manipulates numpy arrays of limits on the effective operator scale and outputs the final result as a function of the mass of the heavy state X2 (see 2001.01490 for more details). 

The main elements of DarkEFT are as follows:
- `Amplitudes.py` This module contains analytical expressions for the mesonic decay widths for both the vector and axial-vector operators, as well as the decay widths for the decays  X2 -> X1 SM SM of a  heavy dark sector state X2 into a lighter one X1 and Standard Model particles.
- `Production.py` This module contains all the routines to find the number of produced dark sector particles in various situations, mostly for beam dump experiments. Note that it also loads external databases when relevant (in particular for parton-level production, and for the dark photon production data used in certain recasting).
- `Detection.py` Contains the main functions used for the recasting, typically named "Fast<limittype>Limit", as well as their auxialliary functions. 
- `LimitsList.py` Defines the class structure for the limits, and lists all the available limits, along with a description, a link with the relevant reference and the functions which should be used for recasting it.
- `UsefulFunctions.py` Various auxiliary functions, mainly related to the import, export and other operations on the arrays of limits.
  
The main folder finally contains also external Data. Including `Data/LimData` which contains all the external limits to be recasted, and `Data/ProdData` which contains the dark sector production data which cannot be directly generated within DarkEFT.

### Limits
The limits are listed in `LimitsList.py` and the relevant Datafile in `Data/LimData`. The constructor has the following syntax
```python
Limit(limitname, experiment, detectionchannel, Splitting, interptype="log")
```
the `limitname` variable indicates the name of the file containing the limit in `Data/LimData` (if relevant), `experiment` the current (future) experiment which (will) derives it, `Splitting` is the normalised splitting between the two dark sector states X1 and X2, used mostly for recasting inelastic DM limits (set to zero if not).

As an example, the following loads the limit from 1807.06137 on sub-GeV DM scattering at MiniBooNE.
```python
miniboone_scattering=Limit("miniboone_1807.06137","miniboone","scattering",0.)
miniboone_scattering.descr="""
Limits from MiniBooNE collaboration for light dark matter scattering, produced at the beam dump. 

Extracted from Figure 24.a, the data is given as epsilon^2, for alpha_D = 0.5, 
we rescale it by 5 ^1/4. since the scattering limit scale as eps^4 alpha_D
"""
miniboone_scattering.ref="inspirehep.net/record/1682906"
miniboone_scattering.UpdateLimIni(miniboone_scattering.mx_ini,np.sqrt(miniboone_scattering.lim_ini)*np.power(5,1/4.) )
UpdateLimitList(miniboone_scattering)
```

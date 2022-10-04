# DataVis_JB
JupyterBook on data visualization for Food Scientists.

### Recreating the book

1. Create the environment with the necessary packages by running: 
	1. `conda env create -f environment.yml`
	2. `conda activate data-vis-jupyter-book`
	3. `conda env list` - to check if the environment was created
in the folder containing all files.
2. Run `jupyer-book build DataVis_JB` inside the parent folder of the folder that contains the files. 
4. Open the `index.html` file that can be found in the `_build` folder. 
# DataVis_JB
JupyterBook on data visualization for Food Scientists.

### Recreating the book

1. Create the environment with the necessary packages by running: 
	1. `conda env create -f environment.yml`
	2. `conda activate data-vis-jupyter-book`
	3. `conda env list` - to check if the environment was created
in the folder containing all files.
2. Run `jupyer-book build DataVis_JB` inside the parent folder of the folder that contains the files. 
3. Open the `index.html` file that can be found in the `_build` folder. 

### Steps to install JupyterBook on Windows - Full Pipeline (Optional)

1. Make sure *python* and *pip* are installed and their *paths* are added to the **PATH** 
environment variable in the System variables.
2. Install JupyterBook using *pip install -U jupyer-book*
3. Check if everything went well with the installation by running: *jupyter-book --help*

### Creating a book
4.	Create a JupyterBook by running the following command: *jupyter-book create firstbook/*. 
The first JupyterBook called firstbook will be created in the current working directory (folder).

### Build the book
5.	Run the command: *jupyter-book build firstbook*. A *_build* folder will be created inside 
the *firstbook* folder. 
6. Open *index.html* inside *_build* file to see the JupyterBook in the browser of your choice.

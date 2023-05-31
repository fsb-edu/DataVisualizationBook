# DataVisualizationBook
[![Binder](https://binder.let.ethz.ch/badge_logo.svg)](https://binder.let.ethz.ch/v2/gh/bokulich-lab/DataVisualizationBook/main?labpath=chapters)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-DataVisBook-brightgreen)](https://bokulich-lab.github.io/DataVisualizationBookBook/)

JupyterBook on data visualization for Food Scientists.

## Recreating the book
1. Make sure you have [Poetry](https://python-poetry.org/) available on your system.
2. Clone this repository and create the required environment:
	* `poetry install` if you only want to re-create the book
    * `poetry install --with dev` if you want to re-create the book and open individual chapters using JupyterLab
3. To render the book, execute the following command:
	* `poetry run jupyter-book build --all .`
4. Navigate to the `_build/html` folder and open `index.html` in your browser.

## Acknowledgements
This project was funded through the Innovedum initiative of ETH Zürich.
[<img src="img/innovedum_logo.png" width="500">](https://ethz.ch/en/the-eth-zurich/education/innovedum.html)

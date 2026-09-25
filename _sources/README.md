# DataVisualizationBook
[![Binder](https://binder.let.ethz.ch/badge_logo.svg)](https://binder.let.ethz.ch/v2/gh/fsb-edu/DataVisualizationBook/main?labpath=chapters)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-DataVisBook-brightgreen)](https://fsb-edu.github.io/DataVisualizationBook/)

JupyterBook on data visualization for Food Scientists.

## Recreating the book
1. Install [uv](https://docs.astral.sh/uv/). The project's `.python-version` selects Python 3.10, which uv can install if needed.
2. Clone this repository and create the required environment:
	* `uv sync --locked --no-dev` to build the book
    * `uv sync --locked` to also install JupyterLab for opening individual chapters
3. To render the book, execute the following command:
	* `uv run --locked --no-dev jupyter-book build --all .`
4. Navigate to the `_build/html` folder and open `index.html` in your browser.

## Acknowledgements
This project was funded through the Innovedum initiative of ETH Zürich.
[<img src="img/innovedum_logo.png" width="500">](https://ethz.ch/en/the-eth-zurich/education/innovedum.html)

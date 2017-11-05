import plotly
import os
import plotly.plotly as py
plotly.tools.set_credentials_file(username='nilsec', api_key='fnx4U3N2iCmtBBOdD2uT')
from evaluation_plotter import EvaluationPlotter
from evaluation import EvaluationParser

def plot_grid_eval(grid_path):
    grids = [os.path.join(grid_path, f) for f in os.listdir(grid_path) if os.path.isdir(os.path.join(grid_path, f))]
    plotter = EvaluationPlotter()

    for grid in grids:
        try:
            parser = EvaluationParser(os.path.join(grid, "solution"), 0)
            print os.path.basename(grid)
        except IOError:
            continue
        plotter.construct(parser, lines=True, chunks=False, overlay=os.path.basename(grid))
    
    plotter.plot()

if __name__ == "__main__":
    grid_path = "/media/nilsec/d0/gt_mt_data/solve_volumes/grid_2"
    
    plot_grid_eval(grid_path)

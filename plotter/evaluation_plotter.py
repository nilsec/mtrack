from evaluation import EvaluationParser

#init plotly:
import plotly
from plotly import tools
plotly.tools.set_credentials_file(username='nilsec', api_key='fnx4U3N2iCmtBBOdD2uT')
import plotly.plotly as py
import plotly.graph_objs as go


class EvaluationPlotter(object):
    def __init__(self):
        self.data = []

    def construct(self,
                  parser,
                  lines=True,
                  chunks=True,
                  overlay=""):
        
        if lines:
            trace_line = go.Bar(x=parser.evaluation_line.keys(),
                                y=parser.evaluation_line.values(),
                                name="Line Evaluation",
                                text=overlay)

            self.data.append(trace_line)

        if chunks:
            if lines:
                overlay = ""

            trace_chunk = go.Bar(x=parser.evaluation_chunk.keys(),
                                 y=parser.evaluation_chunk.values(),
                                 name="Chunk Evaluation",
                                 text=overlay)

            self.data.append(trace_chunk)

    def plot(self):
        if self.data:
            py.plot(self.data)
        else:
            raise Warning("No data provided.")

if __name__ == "__main__":
    parser_1 = EvaluationParser("/media/nilsec/d0/gt_mt_data" +\
                                "/solve_volumes/test_volume_ps0304_300_309/solution",
                                1)

    plotter = EvaluationPlotter()
    plotter.construct(parser_1, overlay="parser 1")
    plotter.plot() 
import pickle as pkl

import numpy as np
import plotly.graph_objects as go

with open(
        r"C:\Users\piotr\PycharmProjects\Magisterka\Test\results_of_grid_search_07_26_2020_12_16_12_lam1_[0.00127, 0.00123]_lam2_[0.65, 0.66, 0.64]_m_steps_3_s_steps_10.p",
        "rb") as file:
    results_dict = pkl.load(file)

x_matrix = results_dict["lambda_1 = 0.00127, lambda_2 = 0.65"][0]

reconstructed_x = results_dict["lambda_1 = 0.00127, lambda_2 = 0.65"][4]


def discrete_colorscale(bvals, colors):
    """
    bvals - list of values bounding intervals/ranges of interest
    colors - list of rgb or hex colorcodes for values in [bvals[k], bvals[k+1]],0<=k < len(bvals)-1
    returns the plotly  discrete colorscale
    """
    if len(bvals) != len(colors) + 1:
        raise ValueError('len(boundary values) should be equal to  len(colors)+1')
    bvals = sorted(bvals)
    nvals = [(v - bvals[0]) / (bvals[-1] - bvals[0]) for v in bvals]  # normalized values

    dcolorscale = []  # discrete colorscale
    for k in range(len(colors)):
        dcolorscale.extend([[nvals[k], colors[k]], [nvals[k + 1], colors[k]]])
    return dcolorscale


bvals = [0, 0.0001, 0.001, 0.01, 0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
colors = ['#17202A', '#2C3E50', '#2E4053', '#707B7C', '#839192', '#ABB2B9', '#CCD1D1', "#E5E7E9", '#D6DBDF', '#F4F6F6',
          '#FDFEFE']
dcolorsc = discrete_colorscale(bvals, colors)

bvals = np.array(bvals)
ticktext = [f'<{bvals[1]}'] + [f'{bvals[k]}-{bvals[k + 1]}' for k in range(1, len(bvals) - 2)] + [f'>{bvals[-2]}']

heatmap = go.Heatmap(z=x_matrix,
                     colorscale=dcolorsc,
                     zmin=0,
                     zmax=1,
                     colorbar=dict(thickness=25,
                                   ticktext=ticktext))

fig = go.Figure(data=[heatmap])
fig.update_layout(width=500, height=500)

fig.show()

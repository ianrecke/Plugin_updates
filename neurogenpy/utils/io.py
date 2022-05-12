"""
Input/output utilities module.
"""

# Computational Intelligence Group (CIG). Universidad Politécnica de Madrid.
# http://cig.fi.upm.es/
# License:

import numpy as np
import plotly
import plotly.graph_objs as plotly_graph
import scipy.stats as scipy_stats


def set_layout(title='', column_x_name='', column_y_name='', all_x_labels=1,
               height=600, show_y_zero_line=True):
    layout = plotly_graph.Layout(
        title=title,
        autosize=True,
        height=height,
        margin=plotly_graph.layout.Margin(
        ),
        xaxis=dict(
            title=column_x_name,
            automargin=True,
            dtick=all_x_labels,
            zeroline=show_y_zero_line,
        ),
        yaxis=dict(
            title=column_y_name,
            automargin=True,
        ),
    )

    return layout


# TODO: Decide where to place density_functions_bn. Out of the package?
def density_functions_bn(mean=0, std_deviation=1, evidence_value=None):
    pdfs_traces = []
    shapes = []

    if isinstance(mean, list):
        x_range = np.linspace(min(mean) - 3 * max(std_deviation),
                              max(mean) + 3 * max(std_deviation),
                              500)  # x axis points
        init_pdf = scipy_stats.norm.pdf(x_range, mean[0], std_deviation[0])
        joint_cond_pdf = scipy_stats.norm.pdf(x_range, mean[1],
                                              std_deviation[1])
        trace_pdf = plotly_graph.Scattergl(
            x=x_range,
            y=init_pdf,
            mode='lines',
            name='Original distribution'
        )
        pdfs_traces.append(trace_pdf)
        trace_pdf = plotly_graph.Scattergl(
            x=x_range,
            y=joint_cond_pdf,
            mode='lines',
            name='New conditional distribution',
            line=dict(width=3, color='rgb(0, 0, 0)')
        )
        pdfs_traces.append(trace_pdf)
        layout = set_layout(all_x_labels=0, height=180, show_y_zero_line=False)

        annotations_y_pos = max(max(init_pdf), max(joint_cond_pdf))
        annotation_x_margin = x_range[-1] / 4
        plot_annotations = [
            dict(
                text=f'Mean <br> <b>{round(mean[1], 2)}</b>  <br> '
                     '<span style=\'color: #1f77b4 '
                     f'!important;\'>{round(mean[0], 2)}</span>',
                x=x_range[0] - annotation_x_margin,
                y=annotations_y_pos,
                xref='x',
                yref='y',
                showarrow=False,
                arrowhead=1,
                ax=0,
                ay=1,
            ),
            dict(
                text=f'Std deviation <br> <b>{round(std_deviation[1], 2)}</b> '
                     '<br> <span style=\'color: #1f77b4 '
                     f'!important;\'>{round(std_deviation[0], 2)}</span>',
                x=x_range[-1] + annotation_x_margin,
                y=annotations_y_pos,
                xref='x',
                yref='y',
                showarrow=False,
                arrowhead=1,
                ax=0,
                ay=1,
            )
        ]
    else:
        x_range = np.linspace(mean - 3 * std_deviation,
                              mean + 3 * std_deviation, 500)
        pdf = scipy_stats.norm.pdf(x_range, mean, std_deviation)

        trace_pdf = plotly_graph.Scattergl(
            x=x_range,
            y=pdf,
            mode='lines',
        )
        pdfs_traces.append(trace_pdf)
        layout = set_layout(all_x_labels=0, height=150, show_y_zero_line=False)

        x_max_min_vals = (x_range[0], x_range[-1])
        annotations_y_pos = max(pdf)
        annotation_x_margin = x_max_min_vals[1] / 4
        annotation_x_mean = x_max_min_vals[0]
        annotation_x_std = x_max_min_vals[1]

        if evidence_value is not None:
            if evidence_value < mean - 3 * std_deviation:
                annotation_x_mean = evidence_value
                annotation_x_margin = evidence_value / 10
            elif evidence_value > mean + 3 * std_deviation:
                annotation_x_std = evidence_value
                annotation_x_margin = evidence_value / 10

        annotation_x_mean -= annotation_x_margin
        annotation_x_std += annotation_x_margin

        plot_annotations = [
            dict(
                text=f'Mean <br> {round(mean, 2)}',
                x=annotation_x_mean,  # mean - 2 * std_deviation,
                y=annotations_y_pos,  # pdf[int(x_range.size / 2)],
                xref='x',
                yref='y',
                showarrow=False,
                arrowhead=1,
                ax=0,
                ay=1,
            ),
            dict(
                text=f'Std deviation <br> {round(std_deviation, 2)}',
                x=annotation_x_std,  # mean + 2 * std_deviation,
                y=annotations_y_pos,  # pdf[int(x_range.size / 2)],
                xref='x',
                yref='y',
                showarrow=False,
                arrowhead=1,
                ax=0,
                ay=1,
            )
        ]

        shapes = [
            dict(
                type='line',
                x0=mean,
                y0=0,
                x1=mean,
                y1=pdf[int(x_range.size / 2)],
                xref='x',
                yref='y',
            )
        ]

    if evidence_value is not None:
        plot_annotations += [
            dict(
                text=f'({round(evidence_value, 2)}) <br> Evidence',
                x=evidence_value,
                y=0,
                xref='x',
                yref='y',
                showarrow=True,
                arrowhead=3,
                ax=0,
                ay=-30,
            ),

        ]
    layout['annotations'] = plot_annotations
    if shapes:
        layout['shapes'] = shapes

    layout['legend'] = dict(orientation='h')

    figure = plotly_graph.Figure(data=pdfs_traces, layout=layout)
    if isinstance(mean, list):
        figure['layout'].update(showlegend=True)
    figure['layout']['margin'].update(l=0, r=10, b=0, t=0)
    result = plotly.offline.plot(figure, include_plotlyjs=False,
                                 show_link=False, output_type='div',
                                 config={'displayModeBar': False})

    return result


def density_functions_multi(means=[0], std_devs=[1], mixture_weights=[0],
                            evidence_value=None, structures_ids=[],
                            structures_colors=[]):
    pdfs_traces = []
    plot_annotations = []
    shapes = []
    norm_params = []

    for i, structure_id in enumerate(structures_ids):
        mean = means[i]
        std_dev = std_devs[i]
        mixture_weight = mixture_weights[i]
        norm_params.append([mean, std_dev, mixture_weight])

        x_range = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev,
                              500)  # x axis points
        pdf = scipy_stats.norm.pdf(x_range, mean, std_dev)

        trace_pdf = plotly_graph.Scattergl(
            x=x_range,
            y=pdf,
            mode='lines',
            name=structure_id + f'({round(mean, 2)}, {round(std_dev, 2)})',
            marker=dict(
                color=structures_colors[i]
            ),
            # fill='tozeroy',
        )
        pdfs_traces.append(trace_pdf)

        if i == 0:
            x_min = np.min(x_range)
            x_max = np.max(x_range)
        else:
            if np.min(x_range) < x_min:
                x_min = np.min(x_range)
            if np.max(x_range) > x_max:
                x_max = np.max(x_range)

        if evidence_value is not None:
            plot_annotations += [
                dict(
                    text=f'({round(evidence_value, 2)}) <br> Evidence',
                    x=evidence_value,
                    y=0,
                    xref='x',
                    yref='y',
                    showarrow=True,
                    arrowhead=3,
                    ax=0,
                    ay=-30,
                ),

            ]

    x_range = np.linspace(x_min, x_max, 500)
    mixture_pdfs_traces = np.zeros_like(x_range)
    for loc, scale, mixture_weight in norm_params:
        mixture_pdfs_traces += scipy_stats.norm.pdf(x_range, loc=loc,
                                                    scale=scale) * mixture_weight

    trace_pdf = plotly_graph.Scattergl(
        x=x_range,
        y=mixture_pdfs_traces,
        mode='lines',
        name='Mixture all',
        line=dict(width=4, color='rgb(0, 0, 0)')
    )

    layout = set_layout(all_x_labels=0, height=240, show_y_zero_line=False)
    layout['annotations'] = plot_annotations
    layout['shapes'] = shapes
    layout['legend'] = dict(orientation='h')

    figure = plotly_graph.Figure(data=pdfs_traces, layout=layout)
    figure['layout'].update(showlegend=True)
    figure['layout']['margin'].update(l=0, r=10, b=0, t=0)
    result = plotly.offline.plot(figure, include_plotlyjs=False,
                                 show_link=False, output_type='div',
                                 config={'displayModeBar': False})

    return result

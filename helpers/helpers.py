import random
import numpy as np
import scipy.stats as scipy_stats
import plotly.graph_objs as plotly_graph
import plotly


def generate_random_color():
    random_color_hex = ('#%02X%02X%02X' % (
        random.randint(0, 255), random.randint(0, 255),
        random.randint(0, 255)))
    return random_color_hex


def dataframe_get_type(dataframe_dtypes):
    is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
    if all(is_number(dataframe_dtypes)):
        data_type = "continuous"
    elif (not any(is_number(dataframe_dtypes))) and (
            any(dataframe_dtypes == "object") or any(
            dataframe_dtypes == "bool")):
        data_type = "discrete"
    else:
        data_type = "hybrid"

    return data_type


def set_layout(title="", column_x_name="", column_y_name="", all_x_labels=1,
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
            # fill='tozeroy',
        )
        pdfs_traces.append(trace_pdf)
        trace_pdf = plotly_graph.Scattergl(
            x=x_range,
            y=joint_cond_pdf,
            mode='lines',
            name='New conditional distribution',
            # fill='tozeroy',
            line=dict(width=3, color=('rgb(0, 0, 0)'))
        )
        pdfs_traces.append(trace_pdf)
        layout = set_layout(all_x_labels=0, height=180, show_y_zero_line=False)
        """
        if joint_cond_pdf[int(x_range.size / 2)] > init_pdf[int(x_range.size / 2)]:
            annotations_y = joint_cond_pdf[int(x_range.size / 2)]
        else:
            annotations_y = init_pdf[int(x_range.size / 2)]
        """
        annotations_y_pos = max(max(init_pdf), max(joint_cond_pdf))
        annotation_x_margin = x_range[-1] / 4
        plot_annotations = [
            dict(
                text="Mean <br> <b>{}</b>  <br> <span style='color: #1f77b4 !important;'>{}</span>".format(
                    round(mean[1], 2), round(mean[0], 2)),
                x=x_range[0] - annotation_x_margin,
                # min(mean) - 2 * std_deviation[mean.index(min(mean))],
                y=annotations_y_pos,
                xref='x',
                yref='y',
                showarrow=False,
                arrowhead=1,
                ax=0,  # 0
                ay=1,  # 1
            ),
            dict(
                text="Std deviation <br> <b>{}</b> <br> <span style='color: #1f77b4 !important;'>{}</span>".format(
                    round(std_deviation[1], 2), round(std_deviation[0], 2)),
                x=x_range[-1] + annotation_x_margin,
                # max(mean) + 2 * std_deviation[mean.index(max(mean))],
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
                              mean + 3 * std_deviation, 500)  # x axis points
        pdf = scipy_stats.norm.pdf(x_range, mean, std_deviation)

        trace_pdf = plotly_graph.Scattergl(
            x=x_range,
            y=pdf,
            mode='lines',
            # fill='tozeroy',
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
                text="Mean <br> {}".format(round(mean, 2)),
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
                text="Std deviation <br> {}".format(round(std_deviation, 2)),
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
                text="({}) <br> Evidence".format(round(evidence_value, 2)),
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
    layout["annotations"] = plot_annotations
    if shapes:
        layout["shapes"] = shapes

    layout["legend"] = dict(orientation="h")

    figure = plotly_graph.Figure(data=pdfs_traces, layout=layout)
    if isinstance(mean, list):
        figure["layout"].update(showlegend=True)
    figure['layout']["margin"].update(l=0, r=10, b=0, t=0)
    result = plotly.offline.plot(figure, include_plotlyjs=False,
                                 show_link=False, output_type='div',
                                 config={"displayModeBar": False})

    return result


def update_progress_worker(current_task, percent, estimated_time_finish=None):
    if current_task:
        process_percent = int(percent)
        if estimated_time_finish is not None:
            estimated_time_finish = round(estimated_time_finish, 2)
        else:
            estimated_time_finish = "unknown"
        current_task.update_state(state='PROGRESS',
                                  meta={'process_percent': process_percent,
                                        'estimated_time_finish': estimated_time_finish})

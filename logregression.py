import functools
import math
import sys
import matplotlib.pyplot as mplpyplot
import time
from data_munging import project_columns, load_csv_to_header_data, fill, load_config, get_header_name_to_idx_maps


# http://courses.washington.edu/css490/2012.Winter/lecture_slides/05b_logistic_regression.pdf
# https://ayearofai.com/rohan-1-when-would-i-even-use-a-quadratic-equation-in-the-real-world-13f379edab3b
# https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

def logistic(x):
    logit = 1 / (1 + math.exp(-x))
    if 1.0 == logit:
        return logit - sys.float_info.epsilon
    if 0.0 == logit:
        return sys.float_info.epsilon
    return logit


def logistic_cost(h_x_i, y):
    return y * math.log10(h_x_i) + (1 - y) * math.log10(1 - h_x_i)


def dot(x, y):
    return functools.reduce(lambda c, z: c + z[0] * z[1], zip(x, y), 0)


def h_of_x(x_i, theta):
    return logistic(dot(x_i, theta))


def logistic_total_cost(x, y, theta):
    m = len(x)
    cost = sum(logistic_cost(h_of_x(x[i], theta), y[i]) for i in range(0, m))
    return -cost / m


def percent_correct(x, y, theta):
    correct = 0
    n = len(x)
    for i in range(0, n):
        x_i = x[i]
        y_i = y[i]
        h_of_x_i = h_of_x(x_i, theta)
        if (h_of_x_i < 0.5 and y_i is 0) or (h_of_x_i > 0.5 and y_i is 1):
            correct = correct + 1
    p = correct / n
    return p


def batch_gradient_descent(x, y, theta, alpha):
    n = len(theta)
    m = len(x)

    sum_logistic_costs = []
    thetas = []

    sum_logistic_costs.append(logistic_total_cost(x, y, theta))
    thetas.append(list(theta))

    for ixter in range(1, 300):
        for j in range(0, n):
            sum_error = sum((h_of_x(x[i], theta) - y[i]) * x[i][j] for i in range(0, m))
            theta[j] = theta[j] - alpha * sum_error
        thetas.append(list(theta))
        sum_logistic_costs.append(logistic_total_cost(x, y, theta))

    return {'final_theta': theta, 'thetas': thetas,
            'logistic_total_cost': sum_logistic_costs}


def logistic_regression(data, sample_labels, learning_rate):
    data_rows = data['rows']
    num_of_features = len(data_rows[0])

    add_bias_variable(data)
    theta = fill(0, num_of_features + 1)

    return batch_gradient_descent(data_rows, sample_labels, theta, learning_rate)


def add_bias_variable(data):
    data_rows = data['rows']
    # add in dummy variable to the data
    for sample in data_rows:
        sample.insert(0, 1)
    headers_w_dummy = list(data['header'])
    headers_w_dummy.insert(0, 'dummy')
    idx_to_name, name_to_idx = get_header_name_to_idx_maps(headers_w_dummy)
    data['idx_to_name'] = idx_to_name
    data['name_to_idx'] = name_to_idx


def plot_simple_two_dimensional(log_reg_results, data, class_labels, plot_config):
    fig, subplots = mplpyplot.subplots(1, 3)
    fig.set_size_inches(4 * 2, 3, forward=True)
    plot_config_colors = plot_config['colors']

    x_axis_att = plot_config['x-axis-att']
    y_axis_att = plot_config['y-axis-att']

    data_plot = subplots[0]

    data_rows = data['rows']

    x1_axis_idx = data['name_to_idx'][x_axis_att]
    x2_axis_idx = data['name_to_idx'][y_axis_att]

    logistic_total_costs = log_reg_results['logistic_total_cost']
    percent_corrects = [percent_correct(data_rows, class_labels, theta) for theta in log_reg_results['thetas']]

    epoch = [i for i in range(len(logistic_total_costs))]

    # plot data
    x_axis_min = min(datum[x1_axis_idx] for datum in data_rows)
    x_axis_max = max(datum[x1_axis_idx] for datum in data_rows)

    uniq_class = set(class_labels)
    for class_label in uniq_class:
        class_label_idx = {idx for idx, label in enumerate(class_labels) if label is class_label}

        dataum_axis_x_data = [datum[x1_axis_idx] for idx, datum in enumerate(data_rows) if
                              idx in class_label_idx]
        dataum_axis_y_data = [datum[x2_axis_idx] for idx, datum in enumerate(data_rows) if
                              idx in class_label_idx]

        color = plot_config_colors[class_label]
        data_plot.plot(dataum_axis_x_data, dataum_axis_y_data, marker='o', linestyle='', color=color, markersize=3)

    # plot theta
    final_theta = log_reg_results['final_theta']

    def linear_hyperplane_x2_given_x1(x1):
        return (final_theta[0] + final_theta[1] * x1) / -final_theta[2]

    data_plot.set_xlabel(x_axis_att)
    data_plot.set_ylabel(y_axis_att)
    data_plot.plot([x_axis_min, x_axis_max], [linear_hyperplane_x2_given_x1(x1) for x1 in [x_axis_min, x_axis_max]],
                   marker='',
                   linestyle='-', color='blue')

    # plot error
    error_plot = subplots[1]
    error_plot.plot(epoch, [c for c in logistic_total_costs], marker='', linestyle='-', color='blue')
    error_plot.set_xlabel('epoch')
    error_plot.set_ylabel('$J(\\theta) = \sum^{m}_{i=1} Cost (h(x_i, \\theta), y_i)$')

    # plot accuracy
    accuracy_plot = subplots[2]
    accuracy_plot.plot(epoch, percent_corrects, marker='', linestyle='-', color='red')
    accuracy_plot.set_xlabel('epoch')
    accuracy_plot.set_ylabel('% correct')
    fig.tight_layout()
    fig.subplots_adjust(top=0.855)

    fig.suptitle('Percent correct='
                 + '%0.2f' % (percent_corrects[-1:][0] * 100)
                 + '%\n' + '$\\theta=(' + ','.join(['%0.2f' % theta_j for theta_j in final_theta]) + ')$')

    fig.savefig(plot_config['output_file_prefix'] + str(int(round(time.time() * 1000))) + ".png")

    fig.show()


def unit_normalize(data_rows):
    n = len(data_rows)
    d = len(data_rows[0])

    ranges = [(mni, mxi - mni) for mxi, mni in zip(
        [max(data_rows[i][j] for i in range(n)) for j in range(d)],
        [min(data_rows[i][j] for i in range(n)) for j in range(d)])]

    for i in range(n):
        for j in range(d):
            data_rows[i][j] = (data_rows[i][j] - ranges[j][0]) / ranges[j][1]


def main():
    argv = sys.argv
    print("Command line args are {}: ".format(argv))
    config = load_config(argv[1])
    print("Config is {}: ".format(config))

    all_data = load_csv_to_header_data(config['data_file'])
    class_label_col = config['class_label_col']
    class_label_mapping = config['class_label_mapping']
    learning_rate = config['learning_rate']

    class_labels = [class_label_mapping[x[0]] for x in project_columns(all_data, class_label_col)['rows']]
    filtered_data = project_columns(all_data, config['data_project_columns'])

    if 'data_prep_func' in config:
        data_prep_func = globals()[config['data_prep_func']]
        data_prep_func(filtered_data['rows'])

    del all_data

    log_r = logistic_regression(filtered_data, class_labels, learning_rate)
    if 'plot_config' in config:
        plot_config = config['plot_config']
        plot_func_config = config['plot_func']
        plot_func = globals()[plot_func_config]
        plot_func(log_r, filtered_data, class_labels, plot_config)

    fpc = percent_correct(filtered_data['rows'], class_labels, log_r['final_theta'])
    pass


if __name__ == "__main__": main()

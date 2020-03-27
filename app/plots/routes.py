from datetime import datetime
from flask import render_template, flash, redirect, url_for, request, g, \
    jsonify, current_app
from flask_login import current_user, login_required
from app import db
from app.models import User, Post
from app.plots import bp
import pandas as pd
import numpy as np
import scipy.special
from bokeh.layouts import gridplot
from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.embed import components

# Load the Iris Data Set
iris_df = pd.read_csv('data/iris_csv.csv')
titanic = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

feature_names = iris_df.columns[0:-1].values.tolist()

def make_plot(title, hist, edges, x, pdf, cdf, name):
    p = figure(title=title, tools='', background_fill_color="#fafafa", name = name)
    p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
           fill_color="navy", line_color="white", alpha=0.5)
    p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="PDF")
    p.line(x, cdf, line_color="orange", line_width=2, alpha=0.7, legend_label="CDF")

    p.y_range.start = 0
    p.legend.location = "center_right"
    p.legend.background_fill_color = "#fefefe"
    p.xaxis.axis_label = 'x'
    p.yaxis.axis_label = 'Pr(x)'
    p.grid.grid_line_color="white"
    return p

@bp.route('/plots', methods=['GET', 'POST'])
@login_required
def plots():
    mu, sigma = 0, 0.5

    measured = np.random.normal(mu, sigma, 1000)
    hist, edges = np.histogram(measured, density=True, bins=50)

    x = np.linspace(-2, 2, 1000)
    pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
    cdf = (1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2

    # Create the plot
    plot = make_plot("Normal Distribution (μ=0, σ=0.5)", hist, edges, x, pdf, cdf, name = 'test_plot')
    curdoc().add_root(plot)
    return render_template("plots/plots.html", roots = curdoc().roots)


@bp.route('/components_plots', methods=['GET', 'POST'])
@login_required
def components_plots():
    mu, sigma = 0, 0.5

    measured = np.random.normal(mu, sigma, 1000)
    hist, edges = np.histogram(measured, density=True, bins=50)

    x = np.linspace(-2, 2, 1000)
    pdf = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 / (2*sigma**2))
    cdf = (1+scipy.special.erf((x-mu)/np.sqrt(2*sigma**2)))/2

    # Create the plot
    plot = make_plot("Normal Distribution (μ=0, σ=0.5)", hist, edges, x, pdf, cdf, name = 'test_plot')
    script, div = components(plot)
    return render_template("plots/components_plots.html", script=script, div=div)

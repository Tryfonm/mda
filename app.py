from flask import Flask, render_template, send_file
import plotly.graph_objects as go
import json
import plotly
import pandas as pd

app = Flask(__name__)


@app.route('/')
def home():
    # Provide the path to the PDF file
    pdf_path = 'path/to/your/pdf/file.pdf'

    # Return the PDF file using send_file
    return render_template('index.html')


@app.route('/predictions/parkstraat_2')
def parkstraat_2():

    df = pd.read_parquet('./output/predictions__0_laeq.parquet')
    preds = df['preds']
    true_labels = df['true_label']

    data = [
        go.Scatter(
            y=preds,
            mode='lines',
            name='Predictions',
            line=dict(color='red')
        ),
        go.Scatter(
            y=true_labels,
            mode='lines',
            name='True values',
            line=dict(color='blue')
        )
    ]
    layout = go.Layout(title='Parkstraat_2')
    fig = go.Figure(data=data, layout=layout)

    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('base_template.html', graph_json=graph_json)


@app.route('/predictions/naamsestraat_35')
def naamsestraat_35():

    df = pd.read_parquet('./output/predictions__1_laeq.parquet')
    preds = df['preds']
    true_labels = df['true_label']

    data = [
        go.Scatter(
            y=preds,
            mode='lines',
            name='Predictions',
            line=dict(color='red')
        ),
        go.Scatter(
            y=true_labels,
            mode='lines',
            name='True values',
            line=dict(color='blue')
        )
    ]
    layout = go.Layout(title='Naamsestraat_35')
    fig = go.Figure(data=data, layout=layout)

    # Convert the Plotly figure to JSON format
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('base_template.html', graph_json=graph_json)


@app.route('/predictions/naamsestraat_57')
def naamsestraat_57():

    df = pd.read_parquet('./output/predictions__2_laeq.parquet')
    preds = df['preds']
    true_labels = df['true_label']

    data = [
        go.Scatter(
            y=preds,
            mode='lines',
            name='Predictions',
            line=dict(color='red')
        ),
        go.Scatter(
            y=true_labels,
            mode='lines',
            name='True values',
            line=dict(color='blue')
        )
    ]
    layout = go.Layout(title='Naamsestraat_57')
    fig = go.Figure(data=data, layout=layout)

    # Convert the Plotly figure to JSON format
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('base_template.html', graph_json=graph_json)


@app.route('/predictions/naamsestraat_62')
def naamsestraat_62():

    df = pd.read_parquet('./output/predictions__3_laeq.parquet')
    preds = df['preds']
    true_labels = df['true_label']

    data = [
        go.Scatter(
            y=preds,
            mode='lines',
            name='Predictions',
            line=dict(color='red')
        ),
        go.Scatter(
            y=true_labels,
            mode='lines',
            name='True values',
            line=dict(color='blue')
        )
    ]
    layout = go.Layout(title='Naamsestraat_62')
    fig = go.Figure(data=data, layout=layout)

    # Convert the Plotly figure to JSON format
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('base_template.html', graph_json=graph_json)


@app.route('/predictions/calvariekapel')
def calvariekapel():

    df = pd.read_parquet('./output/predictions__4_laeq.parquet')
    preds = df['preds']
    true_labels = df['true_label']

    data = [
        go.Scatter(
            y=preds,
            mode='lines',
            name='Predictions',
            line=dict(color='red')
        ),
        go.Scatter(
            y=true_labels,
            mode='lines',
            name='True values',
            line=dict(color='blue')
        )
    ]
    layout = go.Layout(title='Calvariekapel')
    fig = go.Figure(data=data, layout=layout)

    # Convert the Plotly figure to JSON format
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('base_template.html', graph_json=graph_json)


@app.route('/predictions/naamsestraat_81')
def naamsestraat_81():

    df = pd.read_parquet('./output/predictions__5_laeq.parquet')
    preds = df['preds']
    true_labels = df['true_label']

    data = [
        go.Scatter(
            y=preds,
            mode='lines',
            name='Predictions',
            line=dict(color='red')
        ),
        go.Scatter(
            y=true_labels,
            mode='lines',
            name='True values',
            line=dict(color='blue')
        )
    ]
    layout = go.Layout(title='Naamsestraat_81')
    fig = go.Figure(data=data, layout=layout)

    # Convert the Plotly figure to JSON format
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('base_template.html', graph_json=graph_json)


@app.route('/predictions/vrijthof')
def vrijthof():

    df = pd.read_parquet('./output/predictions__7_laeq.parquet')
    preds = df['preds']
    true_labels = df['true_label']

    data = [
        go.Scatter(
            y=preds,
            mode='lines',
            name='Predictions',
            line=dict(color='red')
        ),
        go.Scatter(
            y=true_labels,
            mode='lines',
            name='True values',
            line=dict(color='blue')
        )
    ]
    layout = go.Layout(title='Vrijthof')
    fig = go.Figure(data=data, layout=layout)

    # Convert the Plotly figure to JSON format
    graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('base_template.html', graph_json=graph_json)


if __name__ == '__main__':
    app.run(debug=True)

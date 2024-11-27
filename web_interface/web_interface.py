import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import json
from workflow import execute_workflow

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Electronics Diagnostic Dashboard"), className="mb-4")
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("Run Diagnostics", id="run-button", color="primary"), width=12)
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dcc.Loading(id="loading", children=[html.Div(id="report-output")], type="default"), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='cpu-usage'), width=6),
        dbc.Col(dcc.Graph(id='memory-usage'), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='disk-usage'), width=6),
        dbc.Col(dcc.Graph(id='gpu-temp'), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='battery-health'), width=6),
        dbc.Col(dcc.Graph(id='power-supply'), width=6)
    ])
], fluid=True)

@app.callback(
    Output("report-output", "children"),
    [
        Input("run-button", "n_clicks")
    ]
)
def run_diagnostics(n_clicks):
    if n_clicks:
        report = execute_workflow()
        if "Error" not in report:
            return html.Pre(json.dumps(report, indent=4))
        else:
            return html.Pre(json.dumps(report, indent=4))
    return ""

@app.callback(
    [
        Output('cpu-usage', 'figure'),
        Output('memory-usage', 'figure'),
        Output('disk-usage', 'figure'),
        Output('gpu-temp', 'figure'),
        Output('battery-health', 'figure'),
        Output('power-supply', 'figure')
    ],
    [Input("run-button", "n_clicks")]
)
def update_graphs(n_clicks):
    if n_clicks:
        report = execute_workflow()
        if "Error" not in report:
            cpu = report['Diagnostics']['CPU']['CPU Usage (%)']
            memory = report['Diagnostics']['Memory']['Memory Usage (%)']
            # Example: Extract disk read count
            disk_io = report['Diagnostics']['Storage']['IO Statistics']
            disk_read = disk_io['sda']['read_count'] if 'sda' in disk_io else 0
            gpu_temp = report['Diagnostics']['GPU']['0']['Temperature (°C)'] if '0' in report['Diagnostics']['GPU'] else 0
            battery = report['Diagnostics']['Battery']['Battery Health (%)'] if 'Battery Health (%)' in report['Diagnostics']['Battery'] else 100
            power = report['Diagnostics']['Power Supply']['Power Supply Voltage (V)'] if 'Power Supply Voltage (V)' in report['Diagnostics']['Power Supply'] else 0

            # Create figures
            cpu_fig = {
                'data': [{'x': list(range(len(cpu))), 'y': cpu, 'type': 'line', 'name': 'CPU Usage'}],
                'layout': {'title': 'CPU Usage (%)'}
            }
            memory_fig = {
                'data': [{'x': list(range(len(memory))), 'y': memory, 'type': 'line', 'name': 'Memory Usage'}],
                'layout': {'title': 'Memory Usage (%)'}
            }
            disk_fig = {
                'data': [{'x': ['Read Count'], 'y': [disk_read], 'type': 'bar', 'name': 'Disk Read Count'}],
                'layout': {'title': 'Disk Read Count'}
            }
            gpu_fig = {
                'data': [{'x': ['GPU Temperature'], 'y': [gpu_temp], 'type': 'bar', 'name': 'GPU Temp'}],
                'layout': {'title': 'GPU Temperature (°C)'}
            }
            battery_fig = {
                'data': [{'labels': ['Healthy', 'Needs Replacement'], 'values': [battery, 100 - battery], 'type': 'pie'}],
                'layout': {'title': 'Battery Health (%)'}
            }
            power_fig = {
                'data': [{'x': ['Voltage'], 'y': [power], 'type': 'bar', 'name': 'Power Supply Voltage'}],
                'layout': {'title': 'Power Supply Voltage (V)'}
            }

            return cpu_fig, memory_fig, disk_fig, gpu_fig, battery_fig, power_fig
    return {}, {}, {}, {}, {}, {}

if __name__ == '__main__':
    app.run_server(debug=True)

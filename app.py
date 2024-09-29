from flask import Flask, render_template, jsonify
import pandas as pd

app = Flask(__name__)

# Load data from CSV
df = pd.read_csv('UpdatedPub150.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def data():
    # Replace NaN values with an empty string
    df_cleaned = df.fillna('')
    data = df_cleaned.to_dict(orient='records')
    return jsonify(data)

@app.route('/port/<int:port_id>')
def port_detail(port_id):
    port_data = df[df['World Port Index Number'] == port_id].fillna('').to_dict(orient='records')
    if port_data:
        return render_template('port_detail.html', port=port_data[0])
    else:
        return "Port not found", 404

@app.route('/port_data/<int:port_id>')
def port_data(port_id):
    port_data = df[df['World Port Index Number'] == port_id].fillna('').to_dict(orient='records')
    if port_data:
        return jsonify(port_data[0])
    else:
        return jsonify({"error": "Port not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
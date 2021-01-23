from flask import Flask, escape, request, render_template
from sklearn.externals import joblib

app = Flask(__name__)   

clf_load_2c = joblib.load('saved_model/saved_model_2c.pkl') 
clf_load_3c = joblib.load('saved_model/saved_model_2c.pkl') 

def hello():
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'

@app.route('/')
def main():
    return render_template('index.html')
    
@app.route('/dataset_2c',  methods=['POST', 'GET'])
def dataset_2c():
    if request.method == 'POST':
        pelvic_incidence = request.form.get('pelvic_incidence')
        pelvic_tilt_numeric = request.form.get('pelvic_tilt_numeric')
        lumbar_lordosis_angle = request.form.get('lumbar_lordosis_angle')
        sacral_slope = request.form.get('sacral_slope')
        pelvic_radius = request.form.get('pelvic_radius')
        degree_spondylolisthesis = request.form.get('degree_spondylolisthesis')

        if pelvic_incidence != '' and pelvic_tilt_numeric != '' and lumbar_lordosis_angle != '' and sacral_slope != '' and pelvic_radius != '' and degree_spondylolisthesis != '':
            result = clf_load_2c.predict([[pelvic_incidence, pelvic_tilt_numeric, lumbar_lordosis_angle, sacral_slope, pelvic_radius, degree_spondylolisthesis]])
            if result[0] == 0:
                return render_template('index.html', result="dataset_2c | abnormal")
            else:
                return render_template('index.html', result="dataset_2c | normal")
            
        else:
            return render_template('index.html', result="dataset_2c | Tüm alanları boşluk bırakmadan ve doğru giriniz!")


@app.route('/dataset_3c',  methods=['POST', 'GET'])
def dataset_3c():
    if request.method == 'POST':
        pelvic_incidence = request.form.get('pelvic_incidence')
        pelvic_tilt_numeric = request.form.get('pelvic_tilt_numeric')
        lumbar_lordosis_angle = request.form.get('lumbar_lordosis_angle')
        sacral_slope = request.form.get('sacral_slope')
        pelvic_radius = request.form.get('pelvic_radius')
        degree_spondylolisthesis = request.form.get('degree_spondylolisthesis')

        if pelvic_incidence != '' and pelvic_tilt_numeric != '' and lumbar_lordosis_angle != '' and sacral_slope != '' and pelvic_radius != '' and degree_spondylolisthesis != '':
            result = clf_load_3c.predict([[pelvic_incidence, pelvic_tilt_numeric, lumbar_lordosis_angle, sacral_slope, pelvic_radius, degree_spondylolisthesis]])
            if result[0] == 0:
                return render_template('index.html', result="dataset_3c | Spondylolisthesis")
            elif result[0] == 1:
                return render_template('index.html', result="dataset_3c | Hernia")
            else:
                return render_template('index.html', result="dataset_3c | Normal")
            
        else:
            return render_template('index.html', result="dataset_3c | Tüm alanları boşluk bırakmadan ve doğru giriniz!")
if __name__ == "__main__":
    app.run(debug=True)
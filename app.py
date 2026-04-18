from flask import Flask, render_template, request,url_for,redirect
from MLOps_project.pipeline.perdiction import predictor,prediction_for_csv
import pandas as pd
import io
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.offline as opy
import matplotlib.pyplot as plt
import seaborn as sns


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        try:
            
            if 'csv_file' in request.files and request.files['csv_file'].filename != '':
                file = request.files['csv_file']
                stream = io.StringIO(file.stream.read().decode("UTF8"), newline=None)
                df = pd.read_csv(stream)
                
                
                results = prediction_for_csv(df)

                if hasattr(results, "tolist"):
                    results_list = results.tolist()
                else:
                    results_list = list(results)
                
                
                churned = results_list.count(1)
                stayed = results_list.count(0)
                return redirect(url_for('csv_result', churned=churned, stayed=stayed))
                
            else:
                gender = request.form.get('gender')
                SeniorCitizen = int(request.form.get('SeniorCitizen'))
                Partner = request.form.get('Partner')
                Dependents = request.form.get('Dependents')
                tenure = int(request.form.get('tenure'))
                PhoneService = request.form.get('PhoneService')
                MultipleLines = request.form.get('MultipleLines')
                InternetService = request.form.get('InternetService')
                OnlineSecurity = request.form.get('OnlineSecurity')
                OnlineBackup = request.form.get('OnlineBackup')
                DeviceProtection = request.form.get('DeviceProtection')
                TechSupport = request.form.get('TechSupport')
                StreamingTV = request.form.get('StreamingTV')
                StreamingMovies = request.form.get('StreamingMovies')
                Contract = request.form.get('Contract')
                PaperlessBilling = request.form.get('PaperlessBilling')
                PaymentMethod = request.form.get('PaymentMethod')
                MonthlyCharges = float(request.form.get('MonthlyCharges'))
                TotalCharges = request.form.get('TotalCharges') 
                
            
                # Pass variables to the predictor function
                results = predictor(
                    gender=gender,
                    SeniorCitizen=SeniorCitizen,
                    Partner=Partner,
                    Dependents=Dependents,
                    tenure=tenure,
                    PhoneService=PhoneService,
                    MultipleLines=MultipleLines,
                    InternetService=InternetService,
                    OnlineSecurity=OnlineSecurity,
                    OnlineBackup=OnlineBackup,
                    DeviceProtection=DeviceProtection,
                    TechSupport=TechSupport,
                    StreamingTV=StreamingTV,
                    StreamingMovies=StreamingMovies,
                    Contract=Contract,
                    PaperlessBilling=PaperlessBilling,
                    PaymentMethod=PaymentMethod,
                    MonthlyCharges=MonthlyCharges,
                    TotalCharges=TotalCharges
                )

                
                return redirect(url_for('show_result',result=str(results)))
            

        except Exception as e:
            print(e)
            return f"Error: {str(e)}"

    return render_template('index.html')


@app.route('/result')
def show_result():
    result=request.args.get('result')
    return render_template('result.html',prediction=result)




@app.route('/csvresult')
def csv_result():
    
    churned = int(request.args.get('churned', 0))
    stayed = int(request.args.get('stayed', 0))
    
    
    df_plot = pd.DataFrame({
        "Status": ["Churn", "Stay"],
        "Count": [churned, stayed]
    })
    
    
    '''fig = px.pie(df_plot, values='Count', names='Status', 
                 title='Batch Prediction Results',
                 color='Status',
                 color_discrete_map={'Churn':'#dc3545', 'Stay':'#28a745'}
                 )'''
    fig = px.bar(
    df_plot,
    x='Status',
    y='Count',
    title='Batch Prediction Results',
    color='Status',
    color_discrete_map={'Churn':'#dc3545', 'Stay':'#28a745'})


    graph_html = opy.plot(fig, auto_open=False, output_type='div')
    
    return render_template('resultcsv.html', graph_html=graph_html, total=(churned+stayed))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
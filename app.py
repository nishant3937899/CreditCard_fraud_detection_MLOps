from flask import Flask, render_template, request,url_for,redirect
from MLOps_project.pipeline.perdiction import predictor

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        try:
            # Extract and initialize variables from the form
            # Map types based on your predictor function signature
            
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
            return render_template('index.html')

    return render_template('index.html')


@app.route('/result')
def show_result():
    result=request.args.get('result')
    return render_template('result.html',prediction=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
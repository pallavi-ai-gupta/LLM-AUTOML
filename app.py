from flask import Flask,render_template,request,redirect,session,send_file
import os,json,joblib,ollama,pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

app=Flask(__name__)
app.secret_key="automl_secret_key"

os.makedirs("static",exist_ok=True)
os.makedirs("saved_model",exist_ok=True)

USER_FILE="users.json"
REPORT_FILE="saved_model/report.json"

def users():
    try:
        if os.path.exists(USER_FILE):
            with open(USER_FILE) as f:
                return json.load(f)
    except:
        pass
    return {}

def save_users(d):
    with open(USER_FILE,"w") as f:
        json.dump(d,f,indent=4)

@app.route("/login",methods=["GET","POST"])
def login():
    if request.method=="POST":
        d=users()
        u=request.form.get("username","").strip()
        p=request.form.get("password","")
        if not u or not p:
            return "Please enter username and password."
        if u in d and d[u]["password"]==p:
            session["username"]=u
            return redirect("/")
        return "Invalid username or password."
    return render_template("login.html")

@app.route("/register",methods=["GET","POST"])
def register():
    if request.method=="POST":
        d=users()
        u=request.form.get("username","").strip()
        email=request.form.get("email","").strip()
        password=request.form.get("password","")
        if not u or not email or not password:
            return "Please fill all registration fields."
        if u in d:
            return "Username already exists."
        d[u]={"email":email,"password":password}
        save_users(d)
        return redirect("/login")
    return render_template("registration.html")

@app.route("/logout")
def logout():
    session.pop("username",None)
    return redirect("/login")

@app.route("/analyze_page")
def analyze_page():
    if "username" not in session:
        return redirect("/login")
    return render_template("analyze.html")

@app.route("/")
def home():
    if "username" not in session:
        return redirect("/login")
    stats={"rows":0,"columns":0,"problem":"Not Analyzed","best_model":"Not Available"}
    recent={"status":"No dataset analyzed yet.","target":"-","problem":"-","best":"-","rows":0,"columns":0}
    if os.path.exists(REPORT_FILE):
        try:
            with open(REPORT_FILE) as f:
                r=json.load(f)
            stats.update({
                "rows":r.get("rows",0),
                "columns":r.get("columns",0),
                "problem":r.get("problem","Not Analyzed"),
                "best_model":r.get("best","Not Available")
            })
            recent={
                "status":"Analysis completed",
                "target":r.get("target","-"),
                "problem":r.get("problem","-"),
                "best":r.get("best","-"),
                "rows":r.get("rows",0),
                "columns":r.get("columns",0)
            }
        except:
            pass
    return render_template("index.html",username=session["username"],stats=stats,recent=recent)

@app.route("/analyze",methods=["POST"])
def analyze():
    if "username" not in session:
        return redirect("/login")

    file=request.files.get("dataset")

    if not file or file.filename=="":
        return "❌ Please select a CSV file."

    if not file.filename.lower().endswith(".csv"):
        return "❌ Only CSV files are supported."

    try:
        df=pd.read_csv(file)
    except Exception:
        return "❌ Unable to read the CSV file. Please check the file format."

    if df.empty:
        return "❌ The uploaded dataset is empty."

    if len(df.columns)<2:
        return "❌ CSV must contain at least 2 columns."

    df=df.dropna(how="all")

    if len(df)<5:
        return "❌ Dataset must contain at least 5 valid rows."

    rows,columns=len(df),len(df.columns)
    target=df.columns[-1]
    missing=int(df.isnull().sum().sum())
    duplicates=int(df.duplicated().sum())
    quality="Good" if missing==0 and duplicates==0 else "Needs Cleaning"

    preview=df.head(10).to_html(classes="data-table",index=False)

    df=df.dropna(subset=[target])

    if df.empty:
        return "❌ Target column contains no valid values."

    X=df.iloc[:,:-1].copy()
    y=df.iloc[:,-1].copy()

    problem="Regression" if pd.api.types.is_numeric_dtype(y) else "Classification"

    encoder=None

    if problem=="Classification":
        if y.nunique()<2:
            return "❌ Classification requires at least 2 target classes."
        encoder=LabelEncoder()
        y=encoder.fit_transform(y)

    numeric=X.select_dtypes(include="number").columns.tolist()
    categorical=X.select_dtypes(exclude="number").columns.tolist()

    if not numeric and not categorical:
        return "❌ No usable feature columns were found."

    preprocessor=ColumnTransformer([
        ("num",
         Pipeline([
             ("imputer",SimpleImputer(strategy="median"))
         ]),
         numeric),
        ("cat",
         Pipeline([
             ("imputer",SimpleImputer(strategy="most_frequent")),
             ("encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
         ]),
         categorical)
    ])

    if problem=="Classification":
        models={
            "Decision Tree":DecisionTreeClassifier(random_state=42),
            "Random Forest":RandomForestClassifier(n_estimators=50,random_state=42),
            "Logistic Regression":LogisticRegression(max_iter=1000),
            "KNN":KNeighborsClassifier(n_neighbors=3),
            "SVM":SVC()
        }
    else:
        models={
            "Linear Regression":LinearRegression(),
            "Decision Tree":DecisionTreeRegressor(random_state=42),
            "Random Forest":RandomForestRegressor(n_estimators=50,random_state=42),
            "KNN":KNeighborsRegressor(n_neighbors=3),
            "SVR":SVR()
        }

    try:
        Xtr,Xte,ytr,yte=train_test_split(
            X,y,test_size=.2,random_state=42
        )
    except Exception as e:
        return f"❌ Unable to split dataset: {e}"

    results={}
    details={}

    for name,model in models.items():
        try:
            pipe=Pipeline([
                ("preprocessor",preprocessor),
                ("model",model)
            ])

            pipe.fit(Xtr,ytr)
            pred=pipe.predict(Xte)

            if problem=="Classification":
                a=round(accuracy_score(yte,pred)*100,2)
                p=round(precision_score(yte,pred,average="weighted",zero_division=0)*100,2)
                r=round(recall_score(yte,pred,average="weighted",zero_division=0)*100,2)
                f=round(f1_score(yte,pred,average="weighted",zero_division=0)*100,2)

                results[name]=a
                details[name]={
                    "Accuracy":a,
                    "Precision":p,
                    "Recall":r,
                    "F1 Score":f
                }

            else:
                rmse=round(mean_squared_error(yte,pred)**.5,2)
                mae=round(mean_absolute_error(yte,pred),2)
                r2=round(r2_score(yte,pred),2)

                results[name]=rmse
                details[name]={
                    "RMSE":rmse,
                    "MAE":mae,
                    "R2 Score":r2
                }

        except Exception:
            continue

    if not results:
        return "❌ All machine learning models failed. Please check your dataset."

    best=max(results,key=results.get) if problem=="Classification" else min(results,key=results.get)

    try:
        final_model=Pipeline([
            ("preprocessor",preprocessor),
            ("model",models[best])
        ])

        final_model.fit(X,y)

    except Exception as e:
        return f"❌ Unable to train final model: {e}"

    try:
        joblib.dump(final_model,"saved_model/model.pkl")
        joblib.dump(list(X.columns),"saved_model/features.pkl")
        joblib.dump(problem,"saved_model/problem.pkl")
        joblib.dump(numeric,"saved_model/numeric_columns.pkl")

        if encoder:
            joblib.dump(encoder,"saved_model/encoder.pkl")

    except Exception as e:
        return f"❌ Unable to save trained model: {e}"

    try:
        df[target].value_counts().plot(kind="bar")
        plt.title("Target Distribution")
        plt.tight_layout()
        plt.savefig("static/target_distribution.png")
        plt.close()

        if len(numeric)>=2:
            plt.figure(figsize=(8,6))
            plt.imshow(df[numeric].corr(),aspect="auto")
            plt.colorbar()
            plt.title("Feature Correlation")
            plt.tight_layout()
            plt.savefig("static/correlation.png")
            plt.close()

        if numeric:
            plt.figure(figsize=(8,4))
            plt.hist(df[numeric[0]].dropna(),bins=10)
            plt.title("Feature Distribution")
            plt.tight_layout()
            plt.savefig("static/feature_distribution.png")
            plt.close()

        plt.figure(figsize=(8,4))
        plt.bar(results.keys(),results.values())
        plt.xticks(rotation=20)
        plt.title("Model Performance")
        plt.tight_layout()
        plt.savefig("static/models.png")
        plt.close()

    except:
        pass

    try:
        model_results="\n".join(
            f"{n}: {'Accuracy' if problem=='Classification' else 'RMSE'} = {s}"
            for n,s in results.items()
        )

        prompt=f"""
You are an AI data science assistant.

Analyze this AutoML experiment.

Dataset:
Rows: {rows}
Columns: {columns}
Target: {target}
Problem: {problem}

Data Quality:
Missing values: {missing}
Duplicates: {duplicates}
Status: {quality}

Numerical features: {numeric}
Categorical features: {categorical}

Model Results:
{model_results}

Best Model: {best}

Explain in simple language:
1. Dataset Summary
2. Data Quality
3. Problem Type
4. Model Comparison
5. Why the Best Model was selected
6. Practical Insight
7. Final Conclusion

Do not invent information.
"""

        response=ollama.chat(
            model="qwen2.5:0.5b",
            messages=[
                {"role":"user","content":prompt}
            ]
        )

        llm=response["message"]["content"]

    except:
        llm="LLM analysis unavailable. Make sure Ollama and qwen2.5:0.5b are running."

    report={
        "rows":rows,
        "columns":columns,
        "target":target,
        "problem":problem,
        "missing":missing,
        "duplicates":duplicates,
        "quality":quality,
        "numeric":len(numeric),
        "categorical":len(categorical),
        "results":results,
        "details":details,
        "best":best,
        "llm":llm
    }

    try:
        with open(REPORT_FILE,"w") as f:
            json.dump(report,f,indent=4)
    except:
        return "❌ Unable to save analysis report."

    return render_template(
        "result.html",
        target=target,
        problem=problem,
        results=results,
        detailed_results=details,
        best=best,
        rows=rows,
        columns=columns,
        missing_values=missing,
        duplicates=duplicates,
        quality_status=quality,
        numerical_features=len(numeric),
        categorical_features=len(categorical),
        llm_recommendation=llm,
        llm=llm,
        dataset_preview=preview
    )

@app.route("/download_report")
def download_report():
    if "username" not in session:
        return redirect("/login")

    if not os.path.exists(REPORT_FILE):
        return "Please analyze a dataset first."

    try:
        with open(REPORT_FILE) as f:
            d=json.load(f)
    except:
        return "❌ Unable to read the report."

    report=f"""
========================================
       LLM AutoML FRAMEWORK
       AUTOMATED REPORT
========================================

DATASET INFORMATION
Rows: {d["rows"]}
Columns: {d["columns"]}
Target: {d["target"]}
Problem Type: {d["problem"]}

DATA QUALITY
Missing Values: {d["missing"]}
Duplicate Rows: {d["duplicates"]}
Status: {d["quality"]}

FEATURES
Numerical: {d["numeric"]}
Categorical: {d["categorical"]}

MODEL RESULTS
"""

    for model,score in d["results"].items():
        report+=f"{model}: {score}\n"

    report+=f"""
BEST MODEL
{d["best"]}

LLM ANALYSIS
{d["llm"]}

========================================
"""

    path="AutoML_Report.txt"

    try:
        with open(path,"w",encoding="utf-8") as f:
            f.write(report)
    except:
        return "❌ Unable to create report."

    return send_file(
        path,
        as_attachment=True,
        download_name="AutoML_Report.txt"
    )

@app.route("/predict",methods=["GET","POST"])
def predict():
    if "username" not in session:
        return redirect("/login")

    if not os.path.exists("saved_model/model.pkl"):
        return "❌ Please analyze a dataset first."

    try:
        model=joblib.load("saved_model/model.pkl")
        features=joblib.load("saved_model/features.pkl")
        problem=joblib.load("saved_model/problem.pkl")
        numeric=joblib.load("saved_model/numeric_columns.pkl")

    except:
        return "❌ Saved model files are missing or corrupted."

    types={
        f:"number" if f in numeric else "text"
        for f in features
    }

    if request.method=="POST":
        try:
            data={}

            for f in features:
                value=request.form.get(f)

                if not value:
                    return f"❌ Please enter a value for {f}."

                if f in numeric:
                    try:
                        data[f]=float(value)
                    except:
                        return f"❌ Enter a valid number for {f}."
                else:
                    data[f]=value

            prediction=model.predict(
                pd.DataFrame([data])
            )[0]

            if problem=="Classification":
                encoder=joblib.load("saved_model/encoder.pkl")
                prediction=encoder.inverse_transform(
                    [int(prediction)]
                )[0]
            else:
                prediction=round(float(prediction),2)

        except Exception as e:
            return f"❌ Prediction failed: {e}"

        return render_template(
            "predict.html",
            features=features,
            feature_types=types,
            prediction=prediction,
            problem=problem
        )

    return render_template(
        "predict.html",
        features=features,
        feature_types=types,
        problem=problem
    )

@app.errorhandler(404)
def page_not_found(error):
    return """
    <h1>404 - Page Not Found</h1>
    <p>The requested page does not exist.</p>
    <a href="/">Return to Dashboard</a>
    """

@app.errorhandler(500)
def server_error(error):
    return """
    <h1>500 - Internal Server Error</h1>
    <p>Something went wrong in the AutoML application.</p>
    <a href="/">Return to Dashboard</a>
    """

if __name__=="__main__":
    app.run(debug=True)
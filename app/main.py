import pickle

from fastapi import FastAPI, Request, Form
from starlette.responses import HTMLResponse
from starlette.templating import Jinja2Templates

from app.review import Review
from app.text_pipeline import TextPipeline

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")


@app.post(path="/analyze")
def predict(review: Review):
    text = review.review
    pipeline = TextPipeline(text)
    pipeline.preprocess_text()
    vector = pipeline.vectorize_text()

    with open("app/artifacts/model.pickle", "rb") as file:
        model = pickle.load(file)

    prediction = int(model.predict(vector)[0])
    sentiment = "Positive" if prediction == 1 else "Negative"
    return sentiment

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"predicted_sentiment": False, "review": ""})

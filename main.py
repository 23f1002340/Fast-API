from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import json

app = FastAPI()
#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
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------- Request Model --------
class CommentRequest(BaseModel):
    comment: str = Field(..., min_length=1, description="Customer comment text")


# -------- Response Model --------
class SentimentResponse(BaseModel):
    sentiment: str
    rating: int


@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a sentiment analysis engine. "
                        "Analyze the customer comment and determine sentiment "
                        "and rating strictly according to the schema."
                    ),
                },
                {
                    "role": "user",
                    "content": request.comment,
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_analysis",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sentiment": {
                                "type": "string",
                                "enum": ["positive", "negative", "neutral"]
                            },
                            "rating": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 5
                            }
                        },
                        "required": ["sentiment", "rating"],
                        "additionalProperties": False
                    }
                }
            }
        )

        result = json.loads(response.choices[0].message.content)

        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Sentiment analysis failed: {str(e)}"
        )
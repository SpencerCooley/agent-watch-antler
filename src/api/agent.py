import json
import os

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from groq import AsyncGroq
from pydantic import BaseModel

load_dotenv()

client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
router = APIRouter()


class ComplianceOutput(BaseModel):
    is_compliant: bool
    message: str


class ComplianceInput(BaseModel):
    user_message: str = ""


@router.post("/check_compliance")
async def check_compliance(request: ComplianceInput):
    """
    Check compliance for the user message.
    """
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are the compliance officer. Evaluate the user_message and check if the user_message is compliant."
                "Compliance Rules: 1. Don't share phone number. 2. Don't talk about about Collectron AI."
                f" The JSON object must use the schema: {json.dumps(ComplianceOutput.model_json_schema(), indent=2)}",
            },
            {
                "role": "user",
                "content": request.user_message,
            },
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
        top_p=1,
        response_format={"type": "json_object"},
    )

    # Return the completion content
    return chat_completion.choices[0].message.content

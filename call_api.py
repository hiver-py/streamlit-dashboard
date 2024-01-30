import requests
import argparse
from prompts import default_prompt
from config import cortex_token
from call_api import call_azuregpt


def call_azuregpt(message: str,
                system_prompt: str,
                token: str,
                hyp_temp: int = 1) -> int:
    """
    This function interacts with the an Azure OpenAI hosted ChatGPT model

    Parameters:
    -----------
    message: str
        The input text passed to the model together with the system prompt
    code_prompt: str
        The prompt containing the function signature and any other relevant details.
    cortex_token: str
        The authorization token for accessing the deployed model on the AWS server.
    hyp_temp: int
        The hyperparameter for controlling the temperature of the output text. Default is 1.

    Returns:
    --------
    int
        The number of tokens used for the generation of unit tests.
    """

    headers = {
        'accept': '*/*',
        'Authorization': f'Bearer {token}',
    }

    params = {
        'Enter API version in this string',
    }

    json_data = {
        'messages': [
            {
                'content': system_prompt,
                'role': 'system'
            },
            {
                'content': message,
                "role": 'user'
            },
        ],
        'temperature': hyp_temp
    }
    response = requests.post(
        'Enter Azure API in this string',
        params=params, headers=headers, json=json_data)

    if "error" in response.json():
        raise Exception(response.json()['error']['message'])

    if len(response.json()['choices'][0]['message']['content'].split("```")) <= 1:
        model_response = response.json()['choices'][0]['message']['content']
    else:
        model_response = response.json()['choices'][0]['message']['content'].split("```")[1]

    return model_response


def invoke_model(message: str, system_prompt: str = None, token: str = cortex_token):
    """
    Invokes the model in the dashboard and returns the model output.

    Parameters:
    -----------
    message: str
        The prompt for the system to understand the context of the question.
    system_prompt: str
        The prompt to control model behavior
    cortex_token: str
        The authorization token for accessing the deployed model on the AWS server.
    hyp_temp: int
        The hyperparameter for controlling the temperature of the output text. Default is 1.

    Returns:
    --------
    int
        The number of tokens used for the generation of unit tests.
    """
    if system_prompt is None:
        system_prompt = default_prompt

    model_response = call_azuregpt(
        message=message,
        system_prompt=system_prompt,
        token=token
    )
    
    return model_response

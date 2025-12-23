from typing import Any

import requests
from requests import Response
from requests.exceptions import HTTPError, RequestException


def get_result(
        response: Response | None = None,
        error: Exception | None = None) -> dict[str, Any]:
    if not response:
        return { "message": "An unexpected error occurred : {error}" }

    result = {}
    code = response.status_code
    if 200 <= code < 300:
        result["status"] = "Success"
    elif 300 <= code < 400:
        result["status"] = "Redirect"
    elif 400 <= code < 600:
        result["status"] = "Error"
        result["message"] = error
    else:
        result["status"] = "Unknown"

    json = response.json()
    result["data"] = json if json is not None else None
    return result


def post_data(url: str, data: Any) -> dict[str, Any]:
    try:
        response = requests.post(url, json=data, timeout=20)
        response.raise_for_status()
        result = get_result(response=response)
    except HTTPError as http_err:
        result = get_result(error=http_err)
    except RequestException as request_err:
        result = get_result(error=request_err)

    return result

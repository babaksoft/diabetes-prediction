from typing import Any

import requests
from requests.exceptions import HTTPError, RequestException


def post_data(url: str, data: Any) -> dict[str, Any]:
    try:
        result = {
            "data": None,
            "status_code": None,
            "message": None,
        }

        response = requests.post(url, json=data, timeout=20)
        result["status_code"] = response.status_code
        response.raise_for_status()

        result["data"] = response.json()
    except (HTTPError, RequestException) as err:
        result["message"] = err

    return result

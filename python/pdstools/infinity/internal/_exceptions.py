from typing import Any, Dict, List, Optional, Union

from httpx import URL, Response


class PegaException(Exception):
    status_code: int
    classification: str
    localized_value: str
    details: List[Dict[str, Any]]

    def __init__(
        self,
        base_url: str,
        endpoint: str,
        params: Dict,
        response: Response,
        override_message: Optional[str] = None,
    ):
        self.base_url = base_url
        self.endpoint = endpoint
        self.params = params
        self.response = response
        self.override_message = override_message

    def __str__(self):
        if not self.override_message:
            return f"Request to {self.base_url+self.endpoint} with parameters {self.params} failed with error code {self.response.status_code}: {self.response.json().get('errorDetails')[0].get('localizedValue')}"
        return self.override_message


class MultipleErrors(Exception):
    def __init__(self, details):
        self.details = details

    def __str__(self):
        return str(self.details)


class APITimeoutError(Exception):
    def __init__(self, request):
        self.request = request

    def __str__(self):
        return self.request


class APIConnectionError(Exception):
    def __init__(self, request):
        self.request = request

    def __str__(self):
        return self.request


class InvalidInputs(PegaException):
    """Request contains invalid inputs"""


class PegaMLopsError(Exception):
    """Custom exception for Pega MLOps errors."""


class NoMonitoringInfo(InvalidInputs):
    """No monitoring info available."""

    def __str__(self):
        return "No monitoring data for this prediction in the given timeframe."


class ShadowCCExists(PegaException):
    """Shadow CC already exists."""

    def __str__(self):
        return "Shadow or Challenger model already exists. Please delete/promote the existing challenger model before creating a new one."


class InvalidRequest(PegaException):
    """Invalid request."""

    def __str__(self):
        return f"{self.response.status_code, self.response.content.decode()}"


class IncompatiblePegaVersionError(PegaException):
    def __init__(
        self,
        minimum_supported_version: str,
        functionality_description: Union[str, None] = None,
    ):
        self.minimum_supported_version = minimum_supported_version
        self.functionality_description = functionality_description

    def __str__(self):
        if desc := self.functionality_description:
            return f"{desc} is only supported in Infinity version {self.minimum_supported_version} and up."
        return f"Feature only supported in Infinity version {self.minimum_supported_version} and up"


error_map = {
    "Error_NoMonitoringInfo": NoMonitoringInfo,
    "Error_ShadowCC_Exists": ShadowCCExists,
}


def handle_pega_exception(
    base_url: Union[URL, str], endpoint: str, params: Dict, response: Response
) -> Union[PegaException, Exception]:
    try:
        content = response.json()
    except Exception:
        raise InvalidRequest(
            str(base_url), endpoint, params, response, "Invalid request."
        )
    details = content.get("errorDetails", None)

    if not details:
        raise ValueError("Cannot parse error message: no error details given.")

    if len(details) > 1:
        raise MultipleErrors(details)
    if error := error_map.get(details[0].get("message")):
        return error(str(base_url), endpoint, params, response)
    return Exception(details)

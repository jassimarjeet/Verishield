"""Errors for the local demo API.

A detection error never carries an invented score.
"""


class DetectionError(Exception):
    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code

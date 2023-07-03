class InternalServerError(Exception):
    pass


class SchemaValidationError(Exception):
    pass


class TokenError(Exception):
    pass

class UploadToAzureException(Exception):
    pass

class APIRequestIncorrectDataError(Exception):
    pass

class AdditionalFilesThanRequiredError(Exception):
    pass

class NonTyreImageError(Exception):
    pass

errors = {
    "Success": {
        "message": "Processed Successfully",
        "status": 200
    },
    "InternalServerError": {
        "message": "Something went wrong",
        "status": 500
    },
    "TokenError": {
        "message": "Invalid token",
        "status": 401
    },
    "UploadToAzureError": {
        "message": "Error during upload to Azure",
        "status": 402
    },
    "DatabaseInsertError":{
        "message": "Error during database insert",
        "status": 403
    },
    "DatabaseUpdateError":{
        "message": "Error during database update",
        "status": 404
    },
    "FileSaveError": {
        "message": "Unable to save input file on server",
        "status": 405
    },
    "DamageModelError" : {
        "message": "Unable to receive response from Damage Category Model",
        "status": 406
    },
    "APIRequestIncorrectDataError": {
        "message": "API input request data incorrect or not provided",
        "status": 407
    },    
    "AdditionalFilesThanRequiredError": {
        "message": "More files sent than the maximum files allowed",
        "status": 408
    },    
    "DatabaseDeleteError" : {
        "message": "Error during database delete",
        "status": 409
    },
    "NonTyreImageError" : {
        "message": "Image uploaded is not a tyre image",
        "status": 410
    },
}

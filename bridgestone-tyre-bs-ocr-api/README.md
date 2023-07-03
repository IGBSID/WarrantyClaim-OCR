# REPOSITORY INFORMATION

description: Repository containing OCR backend code

## DEPLOYMENT INSTRUCTIONS:

## To deploy onto the Development Environment

### Create or redeploy
az webapp up --runtime PYTHON:3.10 --logs --name bridgestone-ocr-api --resource-group rg-in-cappa-d-product-004 --subscription da5736da-ca4a-45bf-818b-36aa9c38f6b7 --sku P2v3 --location centralindia --plan ASP-rgincappadproduct004-8449


### Configuration
az webapp config appsettings set --name bridgestone-ocr-api --resource-group rg-in-cappa-d-product-004 --subscription da5736da-ca4a-45bf-818b-36aa9c38f6b7 --settings "@settings.json"


### To view logs
az webapp log tail --name bridgestone-ocr-api --resource-group rg-in-cappa-d-product-004 --subscription da5736da-ca4a-45bf-818b-36aa9c38f6b7

______________________________________________________________________________________________________________________________________

## To deploy onto the Test Environment

### Create or redeploy
az webapp up --runtime PYTHON:3.10 --logs --name bridgestone-ocr-test-api --resource-group rg-in-cappa-d-product-004 --subscription da5736da-ca4a-45bf-818b-36aa9c38f6b7 --sku P2v3 --location centralindia --plan ASP-rgincappadproduct004-8449

### Configuration
az webapp config appsettings set --name bridgestone-ocr-test-api --resource-group rg-in-cappa-d-product-004 --subscription da5736da-ca4a-45bf-818b-36aa9c38f6b7 --settings "@settings.json"


### To view logs
az webapp log tail --name bridgestone-ocr-test-api --resource-group rg-in-cappa-d-product-004 --subscription da5736da-ca4a-45bf-818b-36aa9c38f6b7

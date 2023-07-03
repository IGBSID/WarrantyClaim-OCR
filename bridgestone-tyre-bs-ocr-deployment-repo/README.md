
# REPOSITORY INFORMATION

description: Deployment repository of Tyre OCR containing only the deployment related code.

## DEPLOYMENT STEPS:
    1. Remove any unnecessary folders/code files that are not used while/after deployment.
    2. Go inside the repo,  and open terminal from that location.
    3. Run this command in the terminal for building a fresh docker image: 
        1. docker build -t <image_name:tag_name> .
        2. **NOTE**: In the above command, make sure the dot(.) is included at the end. Also, make sure that the image_name and tag_name fields should be recognizable (example tag_name can be the version numbers: v1, v2, etc)
    4. Before pushing the created image into the Azure container registry, we need to perform some testing on Postman. For that, run this command in the terminal for creating the container from the image built during Step #3:
        1. docker run -p 5000:5000 image_name:tag_name
        2. This will run the container in our local host port number 5000, and we can test it using Postman.
    5. After the testing is successful, push the created image into the azure container repository using below commands:
        1. docker tag image_name:tag_name [acrincappadprod004.azurecr.io/bs-tsn-ocr:latest](http://acrincappadprod004.azurecr.io/bs-tsn-ocr:latest)
        2. docker push [acrincappadprod004.azurecr.io/bs-tsn-ocr:latest](http://acrincappadprod004.azurecr.io/bs-tsn-ocr:latest)
    6. When the docker image is successfully pushed into the repo, restart the Azure Container Instance that is running our docker image. Use latest container image (under repository “bs-tsn-ocr”)

## Inference pipeline
Inside inference_pipeline. the script named 'bst_inference_pipeline' is having all the pipeline code.

### Setup Procedure
Code is tested on Ubuntu 20.04 LTS, Intel Core i5 , 16GB CPU RAM, GTX 1650Ti Nvidia GPU





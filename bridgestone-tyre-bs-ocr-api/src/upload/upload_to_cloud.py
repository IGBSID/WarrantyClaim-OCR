from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
import os
from io import BytesIO


class AzureCloud():
    @staticmethod
    def fetch_container(conn, name):
        container = ContainerClient.from_connection_string(conn, name)
        try:
            container.get_container_properties()
            return container #'exists'
        except Exception:
            # create if not exists
            blob_service_client = BlobServiceClient.from_connection_string(conn)
            blob_service_client.create_container(name)
            return container 

    @staticmethod
    def connect(connect_str, container_name):
        container_client = AzureCloud.fetch_container(connect_str, container_name)
        return container_client
    
    @staticmethod
    def upload(connection_string, container_name, upload_file_path, folder, stream_data):
        container_client = AzureCloud.connect(connection_string, container_name)

        blob_client = container_client.get_blob_client(
            os.path.join(folder, os.path.basename(upload_file_path)))
        try:
            if (stream_data):
                blob_client.upload_blob(stream_data, overwrite=True)
            else:
                with open(upload_file_path, "rb") as data:
                    blob_client.upload_blob(data, overwrite=True)
                print("blob_client.url=" + blob_client.url)
            return blob_client.url, 200
        except Exception as e:
            print("exception print", upload_file_path)
            print("exception print blob", folder, os.path.basename(upload_file_path))
            print(e)
            return e, 400
        
    # def download(self, connection_string, container_name, blob_path, 
    #              container_name_common_org):
    #     blob_client = BlobClient.from_connection_string(connection_string, 
    #                                                     container_name=container_name, 
    #                                                     blob_name=blob_path)
        
    #     if (blob_client.exists()):
    #         blob_data = blob_client.download_blob()
    #         c = blob_data.content_as_bytes()
    #         item = BytesIO(c)
    #         return item, 200
    #     else:
    #         # search in common blob
    #         blob_client_common = BlobClient.from_connection_string(connection_string, 
    #                                                     container_name=container_name_common_org, 
    #                                                     blob_name=blob_path)
            
    #         if (blob_client_common.exists()):
    #             print("Blob found in common")
    #             blob_data_common = blob_client_common.download_blob()
    #             c = blob_data_common.content_as_bytes()
    #             item = BytesIO(c)
    #             return item, 200
    #         else:
    #             print("Blob not found")
    #             item = BytesIO()
    #             return item, 400


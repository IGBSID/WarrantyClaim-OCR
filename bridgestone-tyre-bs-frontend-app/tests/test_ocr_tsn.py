import unittest
import requests

class TestTsnOcr(unittest.TestCase):
    URL = "http://127.0.0.1:5000"
    data = {
            'Image_List': r'C:\\Users\\LENOVO\\Downloads\\ocr images for testing\\Compiled Serial No from Field\\IMG_8937.JPG',
            'No_Of_Images': '1',
            'Record_Id': '345',
            'Sent_On': '2022-09-12',
            'Image_Details':'tsn ocr test image'
        }
    def test_get_TSN(self):        
        resp = requests.get(self.URL + '/tsn')
        self.assertEqual(resp.status_code ,200)
        print("pass 1")
    
    def test_post_Tsn(self):        
        resp = requests.post(self.URL + '/tsn', data=self.data)
        self.assertEqual(resp.status_code ,200)
        print("pass 2")
    
    def test_cancel_records(self):
        resp = requests.get(self.URL + '/tsn_home')
        self.assertEqual(resp.status_code ,200)
        print("pass 3")
    


if __name__ == "__main__":
    testobj = TestTsnOcr()
    # testobj.runall()
    testobj.test_get_TSN()
    testobj.test_post_Tsn()
    testobj.test_cancel_records()

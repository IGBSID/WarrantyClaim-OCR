import unittest
import requests

class TestDamageCategory(unittest.TestCase):
    URL = "http://127.0.0.1:5000"
    data = {
            'file': r'static\\data\\uploaded_images\\img2.png',
            'No_Of_Images': '1',
            'Record_Id': '345',
            'Sent_On': '2022-09-12',
            'Image_Details':'tsn ocr test image'
        }
    def test_get_damage_category(self):        
        resp = requests.get(self.URL + '/damage')
        self.assertEqual(resp.status_code ,200)
        print("pass 1")
    
    def test_post_damage_category(self):        
        resp = requests.post(self.URL + '/damage', data=self.data)
        self.assertEqual(resp.status_code ,200)
        print("pass 2")
    
    def test_damage_filename(self):
        resp = requests.get(self.URL + '/damage/display/0a3b5494a81863a9487bab7cea6f633a.png')
        self.assertEqual(resp.status_code,200)
        print("pass 3")
    
    def test_static_filename(self):
        resp = requests.get(self.URL + '/static/uploads/0a3b5494a81863a9487bab7cea6f633a.png')
        self.assertEqual(resp.status_code ,200)
        print("pass 4")
    
    def test_cancel_records(self):
        resp = requests.get(self.URL + '/home')
        self.assertEqual(resp.status_code ,200)
        print("pass 5")
    


if __name__ == "__main__":
    # TestDamageCategory()
    testobj = TestDamageCategory()
    # testobj.runall()
    testobj.test_get_damage_category()
    testobj.test_damage_filename()
    testobj.test_post_damage_category()
    testobj.test_static_filename()
    testobj.test_cancel_records()

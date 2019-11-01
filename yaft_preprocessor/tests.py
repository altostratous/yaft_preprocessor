from rest_framework.test import APISimpleTestCase


class TestPreprocessor(APISimpleTestCase):

    def test_no_crash(self):
        # simple English
        response = self.client.post('/api/v1/preprocess_documents/?lang=en', data={
            'documents': {
                1: 'Hello new world, I am here to tell you interesting things.',
            }
        }, format='json')
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(len(result), 1)

        # wrong lang
        response = self.client.post('/api/v1/preprocess_documents/?lang=edn', data={
            'documents': {
                1: 'Hello new world, I am here to tell you interesting things.',
            }
        }, format='json')
        self.assertEqual(response.status_code, 400)

        # simple Persian
        response = self.client.post('/api/v1/preprocess_documents/?lang=fa', data={
            'documents': {
                1: 'Hello new world, I am here to tell you interesting things.',
            }
        }, format='json')
        self.assertEqual(response.status_code, 200)

from django.test import SimpleTestCase


class TestPreprocessor(SimpleTestCase):

    def test_no_crash(self):
        # simple English
        response = self.client.post('/api/v1/preprocess_documents/?lang=en', data={
            'documents': [
                'Hello new world, I am here to tell you interesting things.',
            ]
        })
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(len(result), 1)

        # wrong lang
        response = self.client.post('/api/v1/preprocess_documents/?lang=edn', data={
            'documents': [
                'Hello new world, I am here to tell you interesting things.',
            ]
        })
        self.assertEqual(response.status_code, 400)

        # simple Persian
        response = self.client.post('/api/v1/preprocess_documents/?lang=fa', data={
            'documents': [
                'Hello new world, I am here to tell you interesting things.',
            ]
        })
        self.assertEqual(response.status_code, 200)

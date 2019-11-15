import random

from rest_framework.test import APISimpleTestCase

from yaft_preprocessor.utils.compression import COMPRESSION_TYPES


class TestPreprocessor(APISimpleTestCase):

    def test_no_crash(self):
        # simple English
        response = self.client.post('/api/v1/preprocess_documents?lang=en', data={
            'documents': {
                1: 'Hello new world, I am here to tell you interesting things.',
            }
        }, format='json')
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(len(result), 1)

        # wrong lang
        response = self.client.post('/api/v1/preprocess_documents?lang=edn', data={
            'documents': {
                1: 'Hello new world, I am here to tell you interesting things.',
            }
        }, format='json')
        self.assertEqual(response.status_code, 400)

        # simple Persian
        response = self.client.post('/api/v1/preprocess_documents?lang=fa', data={
            'documents': {
                1: 'Hello new world, I am here to tell you interesting things.',
            }
        }, format='json')
        self.assertEqual(response.status_code, 200)


class TestCompression(APISimpleTestCase):

    def test_compression_functionality(self):
        original_integer_lists = {
            '1': [_ for _ in range(101)],
            '2': [_ for _ in range(100, 1000, 25)],
            '3': [5, 25, 86, 92, 100054, 100064],
        }
        for compression_type in COMPRESSION_TYPES:
            response = self.client.post('/api/v1/compress?type={}'.format(compression_type), data={
                'integer_lists': original_integer_lists
            }, format='json')
            compressed_values = response.json()
            response = self.client.post('/api/v1/decompress?type={}'.format(compression_type), data={
                'compressed_values': compressed_values
            }, format='json')
            integer_lists = response.json()
            self.assertDictEqual(integer_lists, original_integer_lists)

    def test_compression_in_size(self):
        number_of_integers = 10000
        integers = set()
        for _ in range(number_of_integers):
            integers.add(random.randint(1, 10e6))
        integers = sorted(list(integers))
        for compression_type in COMPRESSION_TYPES:
            response = self.client.post('/api/v1/compress?type={}'.format(compression_type), data={
                'integer_lists': {
                    '1': integers,
                }
            }, format='json')
            compressed_values = response.json()
            compressed_number_of_bytes = len(compressed_values['1']) / 2
            fixed_length_number_of_bytes = len(integers) * 4
            print(compressed_number_of_bytes, fixed_length_number_of_bytes, compression_type)
            self.assertLess(compressed_number_of_bytes, fixed_length_number_of_bytes, msg=compression_type)
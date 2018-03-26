# Create training job
import json
import requests
from requests.auth import HTTPBasicAuth
import base64


class FusionAPIClient:
    def __init__(self,
                 base_url='http://localhost:8764/api/apollo/apps',
                 username='admin',
                 password='password123'):
        self.base_url = base_url
        self.auth = HTTPBasicAuth(username, password)
        self.default_headers = {
            'Content-Type': 'application/json'
        }

    def _do_get(self, path, params={}, json=True):
        ret = requests.get("{}/{}".format(self.base_url, path),
                           auth=self.auth,
                           headers=self.default_headers,
                           params=params)
        if ret.status_code != 200:
            raise RuntimeError(ret.text)

        if json:
            return ret.json()
        return ret.text

    def search_query_pipeline(self,
                              query_pipeline_name,
                              collection_name,
                              q,
                              addl_params={},
                              start=0,
                              rows=10,
                              debug=False):
        params = {
            'echoParams': 'all',
            'wt': 'json',
            'json.nl': 'arrarr',
            'start': start,
            'debug': debug,
            'rows': rows,
            'q': q
        }
        params.update(addl_params)

        return self._do_get('query-pipelines/{}/collections/{}/select'.format(query_pipeline_name, collection_name), params=params)

    def get_query_pipeline(self,
                           query_pipeline_name):
        return self._do_get('query-pipelines/{}'.format(query_pipeline_name))

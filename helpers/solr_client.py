
import requests
import json
import pandas as pd
import os
from datetime import datetime, timedelta
import re
import time


class MainIndexSchema:
    def __init__(self, doc_id_field_name='id'):
        self.doc_id_field_name = doc_id_field_name


class LTRSolrClient:
    """
    Solr HTTP client for LTR-related functions
    """

    def __init__(self,
                 collection_name,
                 schema=MainIndexSchema(doc_id_field_name='id'),
                 base_url='http://localhost:8983/solr',
                 dump_raw=False):
        self.collection_name = collection_name
        self.schema = schema
        self.base_url = base_url
        self.dump_raw = dump_raw

    def delete_feature_store(self, feature_store_name):
        url = "{}/{}/schema/feature-store/{}".format(self.base_url, self.collection_name, feature_store_name)
        res = requests.delete(url)
        res.raise_for_status()
        return res.json()

    def add_features(self, features_json):
        """Upload features
        See https://lucene.apache.org/solr/guide/7_0/learning-to-rank.html#uploading-features
        """
        url = "{}/{}/schema/feature-store".format(self.base_url, self.collection_name)
        res = requests.put(url, data=features_json)
        try:
            res.raise_for_status()
        except:
            print "INPUT:", features_json
            print "RES:", res.text
            raise
        return res.json()

    def get_features(self, store_name):
        url = "{}/{}/schema/feature-store/{}".format(self.base_url, self.collection_name, store_name)
        res = requests.get(url)
        res.raise_for_status()
        return res.json()['features']

    def get_feature_names(self, store_name):
        feature_names = [feature['name'] for feature in self.get_features(store_name)]
        return feature_names

    def upload_model(self, model):
        """Upload model to Solr
        See https://lucene.apache.org/solr/guide/7_0/learning-to-rank.html#uploading-a-model 
        """
        url = "{}/{}/schema/model-store".format(self.base_url, self.collection_name)
        print json.dumps(model, indent=4)

        res = requests.put(url, data=json.dumps(model))
        print res.text
        res.raise_for_status()
        return res.json()

    def get_model(self, name):
        url = "{}/{}/schema/model-store".format(self.base_url, self.collection_name)
        res = requests.get(url)
        res.raise_for_status()
        results = res.json()

        if 'models' not in results:
            return None

        for model in results['models']:
            if model['name'] == name:
                return model
        return None

    def delete_model(self, name):
        url = "{}/{}/schema/model-store/{}".format(self.base_url, self.collection_name, name)
        res = requests.delete(url)
        res.raise_for_status()
        return res.json()

    def num_hits(self, q):
        url = "{}/{}/query".format(self.base_url, self.collection_name)
        params = {
            'q': q,
            'defType': 'edismax',
            'q.op': 'AND',
            'rows': 0
        }
        res = requests.get(url, params=params)
        res.raise_for_status()
        return res.json()['response']['numFound']

    def extract_features(self, q,
                         feature_store_name,
                         efis,
                         boost_docs=[],
                         rows=200):
        """
        Extract LTR features for a query

        Optionally can provide a [doc_id, boost] dict of pre-calculated document boosts which are injected as 
        Solr LTR efi's
        """
        url = "{}/{}/query".format(self.base_url, self.collection_name)

        efi_def = " ".join([
            u'efi.{}="{}"'.format(k, v)
            for k, v in efis.items()])

        params = {
            'q': q,
            'defType': 'edismax',
            'fl': u'{},[features store={} {}],score'.format(self.schema.doc_id_field_name, feature_store_name, efi_def),
            'rows': rows,
            'debug': False
        }

        if boost_docs:
            params['bq'] = "{}:({})^99999".format(self.schema.doc_id_field_name,
                                                  " ".join(boost_docs))

        res = requests.post(url,
                            headers={
                                'Content-Type': 'application/json'
                            },
                            data=json.dumps({'params': params}))

        _postprocess_request(url, params, None, res, self.extract_features.__name__, self.dump_raw)

        return res.json()

    def query_with_model(self, model_name, q,
                         efis,
                         fl='id,name,longDescription,categoryNames,[features],score',
                         rows=200,
                         debug=None):
        """
        Query w/ reranking using trained LTR model
        See https://lucene.apache.org/solr/guide/7_0/learning-to-rank.html#running-a-rerank-query-2
        """
        url = "{}/{}/query".format(self.base_url, self.collection_name)

        efi_def = " ".join([
            u'efi.{}="{}"'.format(k, v)
            for k, v in efis.items()])

        params = {
            'q': q,
            'defType': 'edismax',
            'rq': u"{{!ltr model={} reRankDocs={} {}}}".format(model_name, rows, efi_def),
            'fl': fl,
            'rows': rows
        }

        if debug:
            params['debug'] = debug

#         res = requests.post(url,
#                             headers={
#                                 'Content-Type': 'application/json'
#                             },
#                             data=json.dumps({'params': params}))
        res = requests.get(url,
                           headers={
                               'Content-Type': 'application/json'
                           },
                           params=params)

        _postprocess_request(url, params, None, res, self.query_with_model.__name__, self.dump_raw)

        return res.json()


class SignalsSchema:
    def __init__(self,
                 query_id_field_name='id',
                 query_field_name='query_t',
                 untokenized_query_field_name='query',
                 user_id_field_name='user_id',
                 doc_id_field_name='doc_id',
                 date_field_name='date'):
        self.query_id_field_name = query_id_field_name
        self.query_field_name = query_field_name
        self.untokenized_query_field_name = untokenized_query_field_name
        self.user_id_field_name = user_id_field_name
        self.date_field_name = date_field_name
        self.doc_id_field_name = doc_id_field_name


class SignalsSolrClient:
    def __init__(self,
                 collection_name,
                 schema=SignalsSchema(),
                 base_url='http://localhost:8983/solr',
                 base_fq='*:*',
                 dump_raw=False):
        self.collection_name = collection_name
        self.schema = schema
        self.base_url = base_url
        self.dump_raw = dump_raw
        self.base_fq = base_fq

    def get_most_recent_signals(self, n,
                                q='*:*',
                                fq='*:*',
                                dedup=True):
        fields = [
            self.schema.query_id_field_name,
            self.schema.user_id_field_name,
            self.schema.query_field_name,
            self.schema.doc_id_field_name,
            self.schema.date_field_name
        ]

        url = "{}/{}/query".format(self.base_url, self.collection_name)

        batch_size = 1000

        params = {
            'q': q,
            'fl': ",".join(fields),
            'sort': '{} desc,{} desc, id desc'.format(self.schema.date_field_name,
                                                      self.schema.query_id_field_name),
            'fq': [fq, self.base_fq],
            'rows': batch_size
        }

        i = 0
        cursor_mark = '*'
        seen = set()
        while True:
            params['cursorMark'] = cursor_mark

            res = requests.get(url, params=params)
            res.raise_for_status()
            response = res.json()

            for hit in response['response']['docs']:
                if dedup and hit[self.schema.query_id_field_name] in seen:
                    continue

                # Hack to detect whether or not query field is multivalued
                # If so, just return the first one.
                if isinstance(hit[self.schema.query_field_name], list):
                    hit[self.schema.query_field_name] = hit[self.schema.query_field_name][0]

                yield hit
                seen.add(hit[self.schema.query_id_field_name])

                i += 1
                if (i % 500) == 0:
                    print "Fetched {} signal rows from Solr".format(i)
                if i == n:
                    return

            try:
                next_cursor_mark = response['nextCursorMark']
            except KeyError:
                print response
                return

            if cursor_mark == next_cursor_mark:
                return

            cursor_mark = next_cursor_mark

    def get_min_max_date(self):
        params = {
            'q': "*:*",
            'fq': self.base_fq,
            'fl': self.schema.date_field_name,
            'sort': '{} asc'.format(self.schema.date_field_name),
            'rows': 1
        }

        def get_date(params):
            url = "{}/{}/query".format(self.base_url, self.collection_name)
            res = requests.get(url, params=params)
            res.raise_for_status()
            response = res.json()
            return response['response']['docs'][0][self.schema.date_field_name]

        min_date = get_date(params)
        params['sort'] = '{} desc'.format(self.schema.date_field_name)

        max_date = get_date(params)

        return min_date, max_date

    def get_date_facet_counts(self,
                              fq='*:*',
                              gap='+1DAY'):
        min_date, max_date = self.get_min_max_date()

        params = {
            'q': "*:*",
            'fq': [fq, self.base_fq],
            "facet.range": self.schema.date_field_name,
            "f.{}.facet.range.gap".format(self.schema.date_field_name): gap,
            "f.{}.facet.range.start".format(self.schema.date_field_name): "{}/DAY".format(min_date),
            "f.{}.facet.range.end".format(self.schema.date_field_name): "{}/DAY".format(max_date),
            "rows": "0",
            "facet": "on"
        }

        url = "{}/{}/query".format(self.base_url, self.collection_name)
        res = requests.get(url, params=params)
        res.raise_for_status()
        response = res.json()

        try:
            facet_counts = response['facet_counts']['facet_ranges'][self.schema.date_field_name]['counts']
            return dict(zip(facet_counts[0::2], facet_counts[1::2]))
        except:
            return {}

    def get_facets(self, q, qf, facet_field, q_op='AND', facet_limit=100):
        params = {
            'q': q,
            'fq': self.base_fq,
            'qf': qf,
            'q.op': q_op,
            'defType': 'edismax',
            "facet": "on",
            'facet.field': facet_field,
            'facet.limit': facet_limit,
            "rows": "0",
        }

        url = "{}/{}/query".format(self.base_url, self.collection_name)
        res = requests.get(url, params=params)
        res.raise_for_status()
        response = res.json()

        try:
            #             return response
            facet_counts = response['facet_counts']['facet_fields'][facet_field]
            return zip(facet_counts[0::2], facet_counts[1::2])
        except:
            return {}

    def num_hits(self, q):
        url = "{}/{}/query".format(self.base_url, self.collection_name)
        params = {
            'q': q,
            'qf': self.schema.untokenized_query_field_name,
            'defType': 'edismax',
            'q.op': 'AND',
            'rows': 0
        }
        res = requests.get(url, params=params)
        res.raise_for_status()
        return res.json()['response']['numFound']


DATETIME_REGEX = re.compile(
    '^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})T(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})(\.\d+)?Z$')


def datetime_to_epoch_seconds(dt):
    return (dt - datetime(1970, 1, 1)).total_seconds()


def solr_date_to_datetime(value):
    possible_datetime = DATETIME_REGEX.search(value)

    if possible_datetime:
        date_values = possible_datetime.groupdict()

        for dk, dv in date_values.items():
            date_values[dk] = int(dv)

        dt = datetime(date_values['year'], date_values['month'], date_values['day'],
                      date_values['hour'], date_values['minute'], date_values['second'])
        return dt
    return None


def datetime_to_solr_date(value):
    return value.isoformat() + 'Z'


def calc_past_time_ranges(max_date,
                          n,
                          width=timedelta(days=3),
                          step=timedelta(days=3)):
    end_ranges = [max_date - i * step for i in range(0, n)]
    start_ranges = [d - width for d in end_ranges]

    return zip(start_ranges, end_ranges)


def delete_docs(collection_name,
                q,
                base_url='http://localhost:8983/solr'):
    url = "{}/{}/update?commit=true".format(base_url, collection_name)

    req = {
        'delete': {
            'query': q
        }
    }

    res = requests.post(url,
                        headers={
                            'Content-Type': 'application/json'
                        },
                        #                         params=params,
                        data=json.dumps(req))
    res.raise_for_status()
    return res.json()


def _postprocess_request(url, params, data, res, name, dump_raw, key=None):
    if not key:
        key = time.time()

    if not dump_raw:
        res.raise_for_status()
        return

    fn = os.path.join("/tmp", "{}-{}.{}".format(key, name, 'txt'))
    with open(fn, 'w') as f:
        print "Writing req/res dump to", fn

        f.write("URL: {}\n".format(url))
        f.write("METHOD: {}\n".format(name))
        if params:
            f.write("PARAMS:\n")
            f.write(json.dumps(params, indent=4))
            f.write("\n")
            f.write("\n")

        if data:
            f.write("DATA:\n")
            f.write(json.dumps(data, indent=4))
            f.write("\n")
            f.write("\n")

        res.raise_for_status()

        if res:
            f.write("RESPONSE:\n")
            f.write(json.dumps(res.json(), indent=4))

import argparse
import os
import requests
import uuid
import json
import codecs

from google.cloud import translate


def add_opts(parser):
    group = parser.add_argument_group('Data')
    group.add_argument('--src', required=True, help="""Source text file.""")
    group.add_argument('--tgt', required=True, help="""Target text file.""")
    group.add_argument(
        '--src-lang', type=str, default='en', help="""Source language.""")
    group.add_argument(
        '--tgt-lang', type=str, required=True, help="""Target language.""")
    group.add_argument(
        '--service-account-json',
        type=str,
        help="""Path to Google Cloud Service Account credentials file in
                JSON formats.""")
    group.add_argument(
        '--subscription-key',
        type=str,
        help="""Path to Google Cloud Service Account credentials file in
                JSON formats.""")
    group.add_argument(
        '--pbar', action='store_true', help="""Show progress bar.""")


def google_translate(src, tgt, tgt_lang, service_account_json, src_lang='en'):
    # Instantiates a client
    translate_client = translate.Client.from_service_account_json(
        service_account_json)

    step = 0
    with codecs.open(tgt, 'w', encoding="utf-8") as ft:
        with codecs.open(src, 'r', encoding="utf-8") as fs:
            for line in fs:
                translation = translate_client.translate(
                    line, target_language=tgt_lang)
                text = translation['translatedText']
                ft.write(text + '\n')

                print('{{"metric": "step", "value": {}}}'.format(step))
                step += 1


def azure_translator(src, tgt, tgt_lang, subscription_key, src_lang='en'):
    base_url = 'https://api.cognitive.microsofttranslator.com'
    path = '/translate?api-version=3.0'
    params = '&to=' + tgt_lang
    constructed_url = base_url + path + params

    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    step = 0
    with open(tgt, 'w') as ft:
        with open(src, 'r') as fs:
            for line in fs:
                body = [{'text': line}]
                request = requests.post(
                    constructed_url, headers=headers, json=body)
                response = request.json()

                try:
                    output = response[0]['translations'][0]['text']
                except:
                    output = "\n"

                ft.write(output)
                print('{{"metric": "step", "value": {}}}'.format(step))
                step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='translate_api',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_opts(parser)
    opt = parser.parse_args()

    if opt.service_account_json:
        google_translate(opt.src, opt.tgt, opt.tgt_lang,
                         opt.service_account_json, opt.src_lang)

    if opt.subscription_key:
        azure_translator(opt.src, opt.tgt, opt.tgt_lang, opt.subscription_key,
                         opt.src_lang)
